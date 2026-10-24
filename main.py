"""Entry point: python main.py [--skip-tuning] [--self-test] [--no-resume] [--config PATH]

Every path is resolved relative to this file, so it works from any working directory. Results are written to
outputs/pipeline_run and checked against the artifacts of eda.ipynb in outputs/ (when present).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import data_preparation as dp  # noqa: E402
import model_training as mt  # noqa: E402

RTOL = 1e-6


def check_baselines(table, ref_dir):
    path = ref_dir / "baseline_results.csv"
    if not path.exists():
        return print("[check] no notebook baselines found, skipping")
    ref = pd.read_csv(path, index_col=0)
    for model in ref.index:
        for col in ("val_accuracy", "val_f1_macro", "val_roc_auc_macro", "val_pr_auc_macro"):
            np.testing.assert_allclose(table.loc[model, col], ref.loc[model, col], rtol=RTOL, atol=1e-9, err_msg=f"baseline {model}.{col}")
    print("[check] baselines reproduce eda.ipynb (outputs/baseline_results.csv)")


def check_tuning(comparison, tuned, ref_dir):
    path, params = ref_dir / "tuned_results.csv", ref_dir / "best_params.json"
    if not (path.exists() and params.exists()):
        return print("[check] no notebook tuning results found, skipping")
    ref, ref_params = pd.read_csv(path, index_col=0), json.loads(params.read_text())
    for model in ref.index:
        np.testing.assert_allclose(comparison.loc[model, "tuned_val_pr_auc"], ref.loc[model, "tuned_val_pr_auc"], rtol=RTOL, atol=1e-9, err_msg=f"tuned {model}")
        assert {k: str(v) for k, v in tuned[model]["best_params"].items()} == {k: str(v) for k, v in ref_params[model].items()}, f"best params differ for {model}"
    print("[check] tuned parameters and validation scores reproduce eda.ipynb")


def check_final(summary, best, ref_dir):
    path = ref_dir / "final_test_results.json"
    if not path.exists():
        return print("[check] no notebook test results found, skipping")
    ref = json.loads(path.read_text())
    assert ref["selected_model"] == best, f"selected model differs: {best} vs {ref['selected_model']}"
    for model, mets in ref["metrics"].items():
        for k, v in mets.items():
            np.testing.assert_allclose(summary["metrics"][model][k], v, rtol=RTOL, atol=1e-9, err_msg=f"test {model}.{k}")
    print("[check] test metrics reproduce eda.ipynb (outputs/final_test_results.json)")


def main():
    parser = argparse.ArgumentParser(description="Activity Level prediction pipeline")
    parser.add_argument("--config", default=str(ROOT / "src" / "config.yaml"))
    parser.add_argument("--skip-tuning", action="store_true", help="stop after the default-parameter baselines")
    parser.add_argument("--self-test", action="store_true", help="run data_preparation self-tests and exit")
    parser.add_argument("--no-resume", action="store_true", help="ignore checkpoints in the output folder and recompute")
    args = parser.parse_args()
    if args.self_test:
        return dp.run_self_tests()

    cfg = dp.load_config(args.config)
    out, ref_dir = dp.resolve(cfg, "outputs"), dp.resolve(cfg, "reference")
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)

    X, y, groups = dp.load_dataset(cfg)
    tr, va, te, seed = dp.group_split(y, groups, cfg["split"]["fractions"], cfg["split"]["seed_search"])
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y.iloc[tr], y.iloc[va], y.iloc[te]
    print(f"rows: {len(X)} | train {len(tr)} val {len(va)} test {len(te)} | session-split seed {seed} | "
          f"sessions {groups.iloc[tr].nunique()}/{groups.iloc[va].nunique()}/{groups.iloc[te].nunique()}")

    baseline, fitted = mt.run_baselines(cfg, X_tr, y_tr, X_va, y_va)
    print("\n== Baselines (library defaults, validation) ==")
    print(baseline.round(4))
    baseline.to_csv(out / "baseline_results.csv")
    check_baselines(baseline, ref_dir)
    if args.skip_tuning:
        return

    tuned, ledger = mt.run_tuning(cfg, X_tr, y_tr, groups.iloc[tr], X_va, y_va, resume=not args.no_resume)
    mt.assert_baselines_intact(cfg, fitted)
    comparison = mt.compare_baseline_tuned(cfg, baseline, tuned)
    print("\n== Baseline vs tuned (validation) ==")
    print(comparison.round(4))
    comparison.to_csv(out / "tuned_results.csv")
    (out / "best_params.json").write_text(json.dumps({n: p["best_params"] for n, p in tuned.items()}, indent=1, default=str))
    check_tuning(comparison, tuned, ref_dir)

    best = comparison["tuned_val_pr_auc"].idxmax()
    print(f"\nSelected by validation {cfg['metrics']['primary']} (pre-declared rule): {best}")
    importance = pd.DataFrame({n: mt.feature_importance(p["model"], X_va, y_va, cfg["seed"], cfg["cleaning"]["drop_columns"],
                                                         cfg["tuning"]["importance_repeats"]) for n, p in tuned.items()})
    importance["mean_rank_score"] = importance.mean(axis=1)
    importance = importance.sort_values("mean_rank_score", ascending=False)
    print("\n== Permutation importance per tuned model (drop in validation macro PR AUC) ==")
    print(importance.round(4))
    importance.to_csv(out / "feature_importance_tuned.csv")

    trva = np.concatenate([tr, va])
    summary, predictions, final_model = mt.final_test(cfg, best, tuned[best]["model"], X.iloc[trva], y.iloc[trva], X_te, y_te,
                                                      ledger, resume=not args.no_resume)
    summary["selected_model"] = best
    summary["best_params"] = {k: str(v) for k, v in tuned[best]["best_params"].items()}
    (out / "final_test_results.json").write_text(json.dumps(summary, indent=1, default=str))
    predictions.insert(0, "session_id", X_te[cfg["group_column"]].to_numpy())
    predictions.to_csv(out / "final_test_predictions.csv")
    joblib.dump(final_model, out / "final_model.joblib")
    print("\n== Test set (evaluated once) ==")
    print(pd.DataFrame(summary["metrics"]).round(4))
    tk = f"{best} (tuned)"
    print("95% CI (tuned):", json.dumps(summary["ci95"][tk]))
    for k, v in summary["paired_improvement"].items():
        print(f"  {k}: {v['mean_diff']:+.4f} CI {np.round(v['ci95'], 4).tolist()}")
    check_final(summary, best, ref_dir)


if __name__ == "__main__":
    main()
