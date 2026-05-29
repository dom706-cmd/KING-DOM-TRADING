#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


ID_COLS = {"symbol", "day", "label", "ts_utc"}


def _load_rr(ds_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(ds_dir, "day=*", "part-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No RR parquet parts found under {ds_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if df.empty:
        raise RuntimeError("RR dataset loaded empty")
    if "label" not in df.columns or "day" not in df.columns:
        raise RuntimeError(f"RR dataset missing label/day; cols={df.columns.tolist()}")
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure base numeric
    for c in out.columns:
        if c in ID_COLS:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Derived features (must also be replicated in live scan later)
    eps = 1e-12

    if "zscore" in out.columns:
        out["abs_zscore"] = out["zscore"].abs()
        out["zscore_sq"] = out["zscore"] * out["zscore"]

    if "tod_min" in out.columns:
        # RTH is 390 minutes; encode cyclic time of day
        ang = (2.0 * math.pi) * (out["tod_min"].clip(0, 390) / 390.0)
        out["tod_sin"] = np.sin(ang)
        out["tod_cos"] = np.cos(ang)

    if "volume" in out.columns:
        out["log_volume"] = np.log1p(out["volume"].clip(lower=0))
    if "transactions" in out.columns:
        out["log_transactions"] = np.log1p(out["transactions"].clip(lower=0))

    if "relvol5" in out.columns and "relvol15" in out.columns:
        out["relvol_ratio"] = out["relvol5"] / (out["relvol15"] + eps)

    if "slope" in out.columns:
        out["abs_slope"] = out["slope"].abs()

    # “how close are we to the band entry”
    if "dist_to_entry_sig" in out.columns:
        out["entry_proximity"] = -out["dist_to_entry_sig"].abs()

    return out


def _time_splits(days: List[date], train_frac: float, val_frac: float) -> Tuple[set, set, set]:
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or (train_frac + val_frac) >= 1:
        raise ValueError("Bad split fractions")
    n = len(days)
    if n < 10:
        raise RuntimeError(f"Too few unique days ({n}) for time-safe train/val/test.")
    i_train = max(1, int(n * train_frac))
    i_val = max(i_train + 1, int(n * (train_frac + val_frac)))
    train_days = set(days[:i_train])
    val_days = set(days[i_train:i_val])
    test_days = set(days[i_val:])
    return train_days, val_days, test_days


def _winsorize_fit(X: pd.DataFrame, q_lo: float = 0.01, q_hi: float = 0.99) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for c in X.columns:
        lo = float(X[c].quantile(q_lo))
        hi = float(X[c].quantile(q_hi))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = -np.inf, np.inf
        # prevent lo>hi from weirdness
        if lo > hi:
            lo, hi = hi, lo
        bounds[c] = (lo, hi)
    return bounds


def _winsorize_apply(X: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = X.copy()
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def _topk_precision_by_day(df: pd.DataFrame, p_col: str, k: int) -> float:
    vals = []
    for d, g in df.groupby("day", sort=True):
        gg = g.sort_values(p_col, ascending=False).head(k)
        if len(gg) == 0:
            continue
        vals.append(float(gg["label"].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--out", default="./models/range_reversion_gold.pkl")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    # tuning
    ap.add_argument("--trials", type=int, default=18)

    # model defaults (baseline)
    ap.add_argument("--max_iter", type=int, default=600)
    ap.add_argument("--learning_rate", type=float, default=0.06)
    ap.add_argument("--max_depth", type=int, default=6)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = _load_rr(args.ds_dir)
    df = _feature_engineer(df)

    # Define features (drop identifiers + leakage)
    feats = [c for c in df.columns if c not in ID_COLS]
    df = df.dropna(subset=feats + ["label"]).copy()

    days = sorted(df["day"].unique())
    train_days, val_days, test_days = _time_splits(days, args.train_frac, args.val_frac)

    train = df[df["day"].isin(train_days)].copy()
    val = df[df["day"].isin(val_days)].copy()
    test = df[df["day"].isin(test_days)].copy()

    if len(train) < 5000 or len(val) < 1000 or len(test) < 1000:
        raise RuntimeError(f"Too few rows after split: train={len(train)} val={len(val)} test={len(test)}")

    X_train = train[feats].astype(float)
    y_train = train["label"].astype(int).to_numpy()
    X_val = val[feats].astype(float)
    y_val = val["label"].astype(int).to_numpy()
    X_test = test[feats].astype(float)
    y_test = test["label"].astype(int).to_numpy()

    # Winsorize (fit on train only)
    clip_bounds = _winsorize_fit(X_train)
    X_train_c = _winsorize_apply(X_train, clip_bounds)
    X_val_c = _winsorize_apply(X_val, clip_bounds)
    X_test_c = _winsorize_apply(X_test, clip_bounds)

    # Class imbalance weights (simple, effective)
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    if pos <= 0:
        raise RuntimeError("No positive labels in train.")
    w_pos = neg / pos
    sample_weight = np.where(y_train == 1, w_pos, 1.0)

    # Random search (real tuning, time-safe using val)
    def sample_params():
        return {
            "learning_rate": float(10 ** rng.uniform(math.log10(0.03), math.log10(0.18))),
            "max_depth": int(rng.integers(3, 10)),
            "max_leaf_nodes": int(rng.integers(15, 80)),
            "min_samples_leaf": int(rng.integers(20, 250)),
            "l2_regularization": float(10 ** rng.uniform(math.log10(1e-4), math.log10(5.0))),
            "max_bins": int(rng.integers(128, 256)),
        }

    best = None
    best_score = -1.0
    best_model = None

    for t in range(args.trials):
        p = sample_params()
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=args.max_iter,
            random_state=args.seed,
            **p,
        )
        model.fit(X_train_c, y_train, sample_weight=sample_weight)

        pv = model.predict_proba(X_val_c)[:, 1]
        pr_auc = average_precision_score(y_val, pv)

        if pr_auc > best_score:
            best_score = float(pr_auc)
            best = p
            best_model = model

        print(f"[TUNE] {t+1:02d}/{args.trials} val_PR_AUC={pr_auc:.4f} params={p}")

    if best_model is None or best is None:
        raise RuntimeError("Tuning failed to produce a model.")

    # Calibrate on validation (time-safe) using isotonic regression on validation probabilities
    p_val = best_model.predict_proba(X_val_c)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)

    # Evaluate on test (calibrated probabilities)
    pt_raw = best_model.predict_proba(X_test_c)[:, 1]
    pt = iso.transform(pt_raw)
    auc = roc_auc_score(y_test, pt)
    pr_auc = average_precision_score(y_test, pt)
    brier = brier_score_loss(y_test, pt)

    test_eval = test.copy()
    test_eval["p"] = pt

    top20 = _topk_precision_by_day(test_eval, "p", 20)
    top50 = _topk_precision_by_day(test_eval, "p", 50)

    bundle = {
        "model": best_model,
        "calibrator": iso,
        "feature_columns": feats,
        "clip_bounds": clip_bounds,
        "tuned_params": best,
        "rows": int(len(df)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "day_min": str(min(days)),
        "day_max": str(max(days)),
        "metrics": {
            "test_auc": float(auc),
            "test_pr_auc": float(pr_auc),
            "test_brier": float(brier),
            "test_top20_precision_by_day": float(top20),
            "test_top50_precision_by_day": float(top50),
            "val_best_pr_auc": float(best_score),
        },
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "feature_engineering": {
            "abs_zscore": True,
            "zscore_sq": True,
            "tod_sin_cos": True,
            "log_volume": True,
            "log_transactions": True,
            "relvol_ratio": True,
            "abs_slope": True,
            "entry_proximity": True,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)

    report_path = os.path.splitext(args.out)[0] + "_report.json"
    with open(report_path, "w") as f:
        json.dump(bundle["metrics"] | {
            "rows": bundle["rows"],
            "train_rows": bundle["train_rows"],
            "val_rows": bundle["val_rows"],
            "test_rows": bundle["test_rows"],
            "day_min": bundle["day_min"],
            "day_max": bundle["day_max"],
            "tuned_params": bundle["tuned_params"],
            "feature_columns": bundle["feature_columns"],
        }, f, indent=2)

    print(f"[OK] saved {args.out}")
    print(f"[OK] report {report_path}")
    print(f"[OK] TEST AUC={auc:.4f} PR_AUC={pr_auc:.4f} BRIER={brier:.5f} TOP20={top20:.4f} TOP50={top50:.4f}")
    print(f"[OK] tuned_params={best}")


if __name__ == "__main__":
    main()
