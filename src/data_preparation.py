"""Data loading, cleaning, feature engineering and the preprocessing Pipeline.

Everything that learns from data (CO2 standardisation constants, log-transform columns, winsorising
quantiles, imputer medians, scaler, one-hot categories) lives inside the Pipeline, so it is fitted on
training rows only. Mirrors the classes in eda.ipynb (Part B).
"""
import sqlite3
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOS_COLUMNS = [f"MetalOxideSensor_Unit{i}" for i in range(1, 5)]


def load_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg, key):
    """Resolve a cfg['paths'][key] entry relative to the project root."""
    return (PROJECT_ROOT / cfg["paths"][key]).resolve()


# --------------------------------------------------------------------------- loading
def normalise_target(series):
    """'Low Activity', 'LowActivity', 'Low_Activity' -> 'Low' (same for Moderate / High)."""
    return series.astype(str).str.replace(r"[\s_]*[Aa]ctivity$", "", regex=True).str.strip()


def load_raw(cfg):
    with sqlite3.connect(resolve(cfg, "db")) as con:
        return pd.read_sql(f"SELECT * FROM {cfg['paths']['table']}", con)


def prepare_rows(df, cfg):
    """Row-level steps that use no statistics: normalise labels, drop exact duplicates (before splitting)."""
    df = df.copy()
    df[cfg["target"]] = normalise_target(df[cfg["target"]])
    if cfg["cleaning"]["drop_duplicates"]:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_dataset(cfg):
    """Return (X, y, groups). X still has the group column; DataCleaner drops it explicitly."""
    df = prepare_rows(load_raw(cfg), cfg)
    return df.drop(columns=[cfg["target"]]), df[cfg["target"]], df[cfg["group_column"]]


# --------------------------------------------------------------------------- transformers
class DataCleaner(BaseEstimator, TransformerMixin):
    """Stateless column-level fixes: K->C, impossible values -> NaN, lower-case HVAC, ordinal codes, missing flags, drop IDs."""

    def __init__(self, drop_columns=("Session ID",), kelvin_threshold=100, humidity_range=(0, 100),
                 co2_infrared_min=0, ordinal_maps=None, add_missing_indicators=True, indicator_columns=()):
        self.drop_columns = drop_columns
        self.kelvin_threshold = kelvin_threshold
        self.humidity_range = humidity_range
        self.co2_infrared_min = co2_infrared_min
        self.ordinal_maps = ordinal_maps
        self.add_missing_indicators = add_missing_indicators
        self.indicator_columns = indicator_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        kelvin = X["Temperature"] > self.kelvin_threshold
        X.loc[kelvin, "Temperature"] = X.loc[kelvin, "Temperature"] - 273.15
        lo, hi = self.humidity_range
        X.loc[(X["Humidity"] < lo) | (X["Humidity"] > hi), "Humidity"] = np.nan
        X.loc[X["CO2_InfraredSensor"] < self.co2_infrared_min, "CO2_InfraredSensor"] = np.nan
        X["HVAC Operation Mode"] = X["HVAC Operation Mode"].astype(str).str.strip().str.lower()
        for col, order in (self.ordinal_maps or {}).items():
            X[col] = X[col].map({v: i for i, v in enumerate(order)}).astype(float)  # unseen level -> NaN
        if self.add_missing_indicators:
            for col in self.indicator_columns:
                X[f"{col}__missing"] = X[col].isna().astype(int)
        return X.drop(columns=[c for c in self.drop_columns if c in X.columns])


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered columns to a CLEANED dataframe. Parameters only in the constructor; statistics learned in fit."""

    CO2 = ["CO2_InfraredSensor", "CO2_ElectroChemicalSensor"]
    MOS_PC = ["MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2", "MetalOxideSensor_Unit4"]
    LOG_CANDIDATES = ["Temperature", "Humidity", "CO2_InfraredSensor", "CO2_ElectroChemicalSensor",
                      "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2", "MetalOxideSensor_Unit3", "MetalOxideSensor_Unit4"]

    def __init__(self, features=("temp_x_humidity", "co2_z_gap"), log_skew_threshold=1.0):
        self.features = features
        self.log_skew_threshold = log_skew_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        if "co2_z_gap" in self.features:
            self.co2_mean_, self.co2_std_ = X[self.CO2].mean(), X[self.CO2].std()
        if "mos_pc1" in self.features:
            self.mos_mean_, self.mos_std_ = X[self.MOS_PC].mean(), X[self.MOS_PC].std()
            self.pca_ = PCA(1, random_state=0).fit(((X[self.MOS_PC] - self.mos_mean_) / self.mos_std_).fillna(0))
        skew = X[self.LOG_CANDIDATES].skew()
        self.log_columns_ = [c for c in skew.index if skew[c] > self.log_skew_threshold]   # learned in fit
        self.log_shift_ = {c: float(X[c].min()) for c in self.log_columns_}
        self.feature_names_out_ = self.feature_names_in_ + list(self.features) + [f"log_{c}" for c in self.log_columns_]
        return self

    def transform(self, X):
        X = X.copy()
        if "temp_x_humidity" in self.features:
            X["temp_x_humidity"] = X["Temperature"] * X["Humidity"] / 100
        if "co2_z_gap" in self.features:
            z = (X[self.CO2] - self.co2_mean_) / self.co2_std_
            X["co2_z_gap"] = z["CO2_ElectroChemicalSensor"] - z["CO2_InfraredSensor"]
        if "mos_pc1" in self.features:
            X["mos_pc1"] = self.pca_.transform(((X[self.MOS_PC] - self.mos_mean_) / self.mos_std_).fillna(0))[:, 0]
        for c in self.log_columns_:
            X[f"log_{c}"] = np.log1p((X[c] - self.log_shift_[c]).clip(lower=0))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to the [q_low, q_high] quantiles learned in fit (NaN-aware)."""

    def __init__(self, quantiles=(0.01, 0.99)):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.low_ = X.quantile(self.quantiles[0]).to_numpy()
        self.high_ = X.quantile(self.quantiles[1]).to_numpy()
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


def _numeric_columns(df, categorical=()):
    return [c for c in df.columns if c not in categorical and not c.endswith("__missing")]


def _flag_columns(df):
    return [c for c in df.columns if c.endswith("__missing")]


def build_preprocessing_pipeline(cfg, drop_columns=None):
    """raw X -> clean -> engineer -> [winsorise -> median impute -> scale | one-hot | missing flags]."""
    c, f = cfg["cleaning"], cfg["features"]
    categorical = tuple(f["categorical"])
    num_pipe = Pipeline([
        ("winsor", Winsorizer(tuple(f["winsorize_quantiles"]))),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    columns = ColumnTransformer(
        [("num", num_pipe, partial(_numeric_columns, categorical=categorical)),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
         ("flags", "passthrough", _flag_columns)],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("clean", DataCleaner(
            drop_columns=tuple(c["drop_columns"] if drop_columns is None else drop_columns),
            kelvin_threshold=c["kelvin_threshold"], humidity_range=tuple(c["humidity_range"]),
            co2_infrared_min=c["co2_infrared_min"], ordinal_maps=c["ordinal_maps"],
            add_missing_indicators=c["add_missing_indicators"], indicator_columns=tuple(c["indicator_columns"]))),
        ("engineer", FeatureEngineer(tuple(f["engineered"]), f["log_skew_threshold"])),
        ("columns", columns),
    ])


# --------------------------------------------------------------------------- splitting
def group_split(y, groups, fractions=(0.70, 0.15, 0.15), n_seeds=300):
    """Split by whole sessions into train/val/test so no session spans two splits.

    Tries n_seeds session shuffles and keeps the one whose row fractions and class mix are closest
    to the targets (selection uses label proportions only, never model performance).
    Returns (train_idx, val_idx, test_idx, chosen_seed) as positional index arrays.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    classes = np.unique(y)
    overall = np.array([(y == k).mean() for k in classes])
    uniq = np.unique(groups)
    sizes = pd.Series(groups).value_counts()
    best = None
    for seed in range(n_seeds):
        order = np.random.RandomState(seed).permutation(uniq)
        assign, cum = {}, 0
        for g in order:
            share = cum / len(y)
            assign[g] = 0 if share < fractions[0] else (1 if share < fractions[0] + fractions[1] else 2)
            cum += sizes[g]
        part_of_row = np.array([assign[g] for g in groups])
        score = 0.0
        for p in range(3):
            m = part_of_row == p
            if m.sum() == 0:
                score = np.inf
                break
            mix = np.array([(y[m] == k).mean() for k in classes])
            score += abs(m.mean() - fractions[p]) * 2 + np.abs(mix - overall).max()
        if best is None or score < best[0]:
            best = (score, seed, part_of_row)
    _, seed, part_of_row = best
    return (*[np.where(part_of_row == p)[0] for p in range(3)], seed)


# --------------------------------------------------------------------------- self-tests
def run_self_tests():
    """Eight essential checks; raises AssertionError on failure."""
    cfg = load_config()
    X, y, g = load_dataset(cfg)
    c = cfg["cleaning"]
    cleaned = DataCleaner(
        drop_columns=tuple(c["drop_columns"]), ordinal_maps=c["ordinal_maps"],
        indicator_columns=tuple(c["indicator_columns"])).transform(X)
    assert "Session ID" not in cleaned.columns, "1: leaky ID column must be dropped"
    assert cleaned["Temperature"].max() < 60, "2: Kelvin rows must be converted"
    assert cleaned["Humidity"].dropna().between(0, 100).all(), "3: impossible humidity must become NaN"
    assert cleaned["Humidity__missing"].sum() > 0, "4: missing indicators must exist"
    assert set(y.unique()) == set(cfg["target_labels"]), "5: target labels must collapse to 3 classes"

    tr, va, te, _ = group_split(y, g, cfg["split"]["fractions"], 30)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)), "6: split index overlap"
    assert len(tr) + len(va) + len(te) == len(X)
    assert not (set(g.iloc[tr]) & set(g.iloc[va]) or set(g.iloc[tr]) & set(g.iloc[te])), "6: session leak"

    pipe = build_preprocessing_pipeline(cfg)
    try:
        pipe.transform(X.iloc[:5])
        raise AssertionError("7: unfitted pipeline must refuse to transform")
    except NotFittedError:
        pass
    Xt = pipe.fit(X.iloc[tr]).transform(X.iloc[te])
    assert not np.isnan(np.asarray(Xt, dtype=float)).any(), "8: no NaN after the pipeline"
    messy = X.iloc[te].copy()
    messy["HVAC Operation Mode"] = "never_seen_mode"
    messy["Time of Day"] = "dawn"
    assert pipe.transform(messy).shape == Xt.shape, "8: unseen categories must not crash"
    print("self-tests passed")


if __name__ == "__main__":
    run_self_tests()
