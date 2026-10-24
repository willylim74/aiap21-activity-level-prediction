"""Model registry, baselines, budgeted tuning with a resumable ledger, importance, one-time test evaluation.

Mirrors eda.ipynb (Parts B and C): baselines are library defaults; tuned settings and class weights are
applied only on clones in the tuning path; grouped CV by session; successive halving on a subsample and a
refit of the winner on the full training split; the test set is evaluated once.
"""
import hashlib
import importlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (activates HalvingRandomSearchCV)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from data_preparation import build_preprocessing_pipeline, resolve


# --------------------------------------------------------------------------- models
def _import(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def build_model(name, cfg, tuned_fixed=False):
    """Full pipeline (preprocessing + classifier). Library defaults unless tuned_fixed=True."""
    spec = cfg["models"][name]
    params = dict(spec.get("params", {}))
    if tuned_fixed:
        params.update(spec.get("tuned_fixed", {}))
    return Pipeline([("prep", build_preprocessing_pipeline(cfg)), ("clf", _import(spec["class"])(**params))])


def tunable_models(cfg):
    return [n for n, s in cfg["models"].items() if s.get("search")]


def assert_baselines_intact(cfg, fitted):
    """Baselines must still equal freshly built library-default estimators (guards against contamination)."""
    for name, model in fitted.items():
        fresh = _import(cfg["models"][name]["class"])(**cfg["models"][name].get("params", {}))
        assert model.named_steps["clf"].get_params() == fresh.get_params(), f"{name}: parameters differ from defaults"


# --------------------------------------------------------------------------- metrics
def pr_auc_macro(y_true, proba, classes):
    """classes MUST match the probability columns (scikit-learn: alphabetical = model.classes_)."""
    onehot = label_binarize(y_true, classes=classes)
    return float(np.mean([average_precision_score(onehot[:, k], proba[:, k]) for k in range(len(classes))]))


def compute_metrics(y_true, proba, classes):
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(classes)),
        "pr_auc_macro": pr_auc_macro(y_true, proba, classes),
    }


def evaluate(model, X, y):
    return compute_metrics(y, model.predict_proba(X), list(model.classes_))


# --------------------------------------------------------------------------- baselines
def run_baselines(cfg, X_tr, y_tr, X_va, y_va):
    """Every model at library defaults on train, scored on validation; table ranked by macro PR AUC."""
    rows, fitted = [], {}
    for name in cfg["models"]:
        t0 = time.time()
        model = build_model(name, cfg).fit(X_tr, y_tr)
        val, train = evaluate(model, X_va, y_va), evaluate(model, X_tr, y_tr)
        rows.append({"model": name, **{f"val_{k}": v for k, v in val.items()}, "train_pr_auc_macro": train["pr_auc_macro"],
                     "gap_pr_auc (train-val)": train["pr_auc_macro"] - val["pr_auc_macro"], "fit_s": round(time.time() - t0, 1)})
        fitted[name] = model
    table = pd.DataFrame(rows).set_index("model").sort_values("val_pr_auc_macro", ascending=False)
    table["rank"] = np.arange(1, len(table) + 1)
    return table, fitted


# --------------------------------------------------------------------------- ledger
class Ledger:
    """JSON ledger + joblib checkpoints keyed 'stage::model'; an entry is reused only if its fingerprint matches.

    Checkpoints only protect an interrupted run; delete the folder (or use --no-resume) to recompute everything.
    """

    def __init__(self, cfg):
        self.dir = resolve(cfg, "outputs")
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "ledger.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def fingerprint(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _ckpt(self, key):
        return self.dir / "checkpoints" / (key.replace("::", "__") + ".joblib")

    def get(self, key, fingerprint):
        entry = self.data.get(key)
        if entry and entry["fingerprint"] == fingerprint and self._ckpt(key).exists():
            return entry, joblib.load(self._ckpt(key))
        return None

    def put(self, key, fingerprint, report, payload, **meta):
        joblib.dump(payload, self._ckpt(key))
        self.data[key] = {"fingerprint": fingerprint, "report": report, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
        self.path.write_text(json.dumps(self.data, indent=1, default=float))


# --------------------------------------------------------------------------- tuning
def _scorer(classes):
    return make_scorer(lambda yt, p: pr_auc_macro(yt, p, classes), response_method="predict_proba")


def _grid_size(space):
    return int(np.prod([len(v) for v in space.values()]))


def tune_model(name, cfg, X_tr, y_tr, groups_tr, ledger, resume=True, log=print):
    """Grid search for the linear model, successive halving (resource = rows) for the rest.

    refit=False inside the search; the winner is then refit on the FULL training split on a clone that
    carries the tuned-path settings (baselines are never touched).
    """
    spec, t = cfg["models"][name], cfg["tuning"]
    key = f"tuned::{name}"
    fp = Ledger.fingerprint({"space": {k: [str(x) for x in v] for k, v in spec["grid"].items()}, "fixed": spec.get("tuned_fixed", {}),
                             "search": spec["search"], "n_train": len(X_tr), "seed": cfg["seed"], "tuning": t, "n_features": X_tr.shape[1]})
    hit = ledger.get(key, fp) if resume else None
    if hit:
        log(f"[checkpoint reused] {key}")
        return hit[1]
    base = build_model(name, cfg, tuned_fixed=True)
    common = dict(scoring=_scorer(sorted(y_tr.unique())), cv=GroupKFold(n_splits=t["cv_folds"]), n_jobs=t["n_jobs"], refit=False)
    if spec["search"] == "grid":
        search = GridSearchCV(base, spec["grid"], **common)
    else:
        search = HalvingRandomSearchCV(
            base, spec["grid"], n_candidates=min(t["n_candidates"], _grid_size(spec["grid"])), factor=t["factor"], resource="n_samples",
            min_resources=t["min_resources"], max_resources=min(t["max_resources"], len(X_tr)), random_state=cfg["seed"],
            aggressive_elimination=False, **common)
    t0 = time.time()
    search.fit(X_tr, y_tr, groups=groups_tr)
    search_s = time.time() - t0
    best_params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in search.best_params_.items()}
    t1 = time.time()
    model = clone(base).set_params(**best_params).fit(X_tr, y_tr)
    cvres = pd.DataFrame(search.cv_results_)
    row = cvres.loc[search.best_index_]
    folds = [float(row[f"split{i}_test_score"]) for i in range(t["cv_folds"])]
    payload = {"model": model, "best_params": best_params, "cv_folds": folds, "search_s": search_s, "refit_s": time.time() - t1}
    report = f"{name}: best {best_params} | CV PR AUC {np.mean(folds):.4f} (fold sd {np.std(folds):.4f}) | search {search_s:.0f}s"
    ledger.put(key, fp, report, payload, best_params=best_params)
    log(report)
    return payload


def run_tuning(cfg, X_tr, y_tr, g_tr, X_va, y_va, resume=True, log=print):
    ledger, tuned = Ledger(cfg), {}
    for name in tunable_models(cfg):
        payload = tune_model(name, cfg, X_tr, y_tr, g_tr, ledger, resume, log)
        payload["val"] = evaluate(payload["model"], X_va, y_va)
        tuned[name] = payload
    return tuned, ledger


def compare_baseline_tuned(cfg, baseline_table, tuned):
    primary = f"val_{cfg['metrics']['primary']}"
    rows = []
    for n, p in tuned.items():
        sd_t = float(np.std(p["cv_folds"]))
        rows.append({"model": n, "baseline_val_pr_auc": baseline_table.loc[n, primary], "tuned_val_pr_auc": p["val"]["pr_auc_macro"],
                     "delta": p["val"]["pr_auc_macro"] - baseline_table.loc[n, primary], "tuned_cv_fold_sd": sd_t,
                     "baseline_f1": baseline_table.loc[n, "val_f1_macro"], "tuned_f1": p["val"]["f1_macro"],
                     "tuned_cv_pr_auc": float(np.mean(p["cv_folds"])), "tuned_val_roc_auc": p["val"]["roc_auc_macro"]})
    return pd.DataFrame(rows).set_index("model").sort_values("tuned_val_pr_auc", ascending=False)


# --------------------------------------------------------------------------- importance
def feature_importance(model, X_va, y_va, seed, drop_columns=(), n_repeats=10):
    """Permutation importance on RAW columns (whole pipeline): drop in macro PR AUC on validation."""
    X_va = X_va.drop(columns=list(drop_columns))
    res = permutation_importance(model, X_va, y_va, scoring=_scorer(sorted(y_va.unique())), n_repeats=n_repeats,
                                 random_state=seed, n_jobs=-1)
    return pd.Series(res.importances_mean, index=X_va.columns)


# --------------------------------------------------------------------------- final test (ONCE)
def _bootstrap(probas, y_true, classes, n, seed):
    """Paired bootstrap of macro PR AUC and macro-F1; the same resample is used for every model."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    out = {m: {"pr": [], "f1": []} for m in probas}
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < len(classes):
            continue
        for m, p in probas.items():
            out[m]["pr"].append(pr_auc_macro(y_true[idx], p[idx], classes))
            out[m]["f1"].append(f1_score(y_true[idx], np.asarray(classes)[p[idx].argmax(1)], average="macro", zero_division=0))
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in out.items()}


def final_test(cfg, best_name, tuned_model, X_trva, y_trva, X_te, y_te, ledger, resume=True, log=print):
    """Refit the chosen tuned model, its default twin and the naive model on train+val; score the test set ONCE."""
    key = "final::test"
    fp = Ledger.fingerprint({"model": best_name, "params": {k: str(v) for k, v in tuned_model.get_params().items() if k.startswith("clf__")},
                             "n_trva": len(X_trva), "n_test": len(X_te), "seed": cfg["seed"]})
    hit = ledger.get(key, fp) if resume else None
    if hit:
        log("test set already evaluated: loading the stored predictions (no new evaluation)")
        payload = hit[1]
    else:
        models = {f"{best_name} (tuned)": clone(tuned_model).fit(X_trva, y_trva),
                  f"{best_name} (default)": build_model(best_name, cfg).fit(X_trva, y_trva),
                  "naive": build_model("naive", cfg).fit(X_trva, y_trva)}
        probas = {k: pd.DataFrame(m.predict_proba(X_te), columns=list(m.classes_), index=X_te.index) for k, m in models.items()}
        payload = {"models": models, "probas": probas}
        ledger.put(key, fp, f"test evaluated once for {best_name}", payload, model=best_name)
    probas = payload["probas"]
    classes = list(next(iter(probas.values())).columns)
    metrics = {k: compute_metrics(y_te, p.to_numpy(), classes) for k, p in probas.items()}
    tuned_key, default_key = f"{best_name} (tuned)", f"{best_name} (default)"
    boot = _bootstrap({k: p.to_numpy() for k, p in probas.items()}, y_te, classes, cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    summary = {"metrics": metrics, "ci95": {k: {"pr_auc_macro": ci(d["pr"]), "f1_macro": ci(d["f1"])} for k, d in boot.items()}, "paired_improvement": {}}
    for other in (default_key, "naive"):
        for m, label in (("pr", "pr_auc_macro"), ("f1", "f1_macro")):
            diff = boot[tuned_key][m] - boot[other][m]
            summary["paired_improvement"][f"tuned minus {other} | {label}"] = {"mean_diff": float(diff.mean()), "ci95": ci(diff)}
    pred = probas[tuned_key].copy()
    pred.insert(0, "predicted", np.asarray(classes)[pred.to_numpy().argmax(1)])
    pred.insert(0, "actual", np.asarray(y_te))
    pred.index.name = "row_id"
    return summary, pred, payload["models"][tuned_key]
