#!/usr/bin/env python3
# train_ab_from_parquet.py
#
# Train Model A (liquid) and Model B (outlier) from day-wise parquet shards produced by
# build_ab_dataset_from_flatfiles.py.
#
# GOLD STANDARD:
# - Real data only (no placeholders)
# - Chronological evaluation by default (time split)
# - Portable artifact saved via pickle (so pickle.load() works)
#
from __future__ import annotations

import argparse
import glob
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [
    "trend_20_50",
    "vol20",
    "avg20_dollar_vol",
    "mom5",
    "mom20",
    "gap_pct",
    "atr14_pct",
    "or_range_pct",
    "relvol5",
    "relvol15",
]


def _load_ds(ds_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(Path(ds_dir) / "day=*" / "part-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet shards found under {ds_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError("Dataset empty after reading shards")
    return df


def _time_split(df: pd.DataFrame, *, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if "day" not in df.columns:
        raise RuntimeError("time split requires 'day' column in dataset")
    days = pd.to_datetime(df["day"], errors="coerce")
    if days.isna().all():
        raise RuntimeError("Could not parse 'day' column to datetime for time split.")
    df = df.copy()
    df["__day_dt"] = days
    # unique days in ascending order
    uniq = np.array(sorted(df["__day_dt"].dt.date.unique()))
    if len(uniq) < 30:
        raise RuntimeError(f"Not enough unique days ({len(uniq)}) for a stable time split.")
    cut = int(np.floor((1.0 - float(test_size)) * len(uniq)))
    cut = min(max(cut, 1), len(uniq) - 1)
    train_days = set(uniq[:cut])
    test_days = set(uniq[cut:])
    tr_idx = np.flatnonzero(df["__day_dt"].dt.date.isin(train_days).values)
    te_idx = np.flatnonzero(df["__day_dt"].dt.date.isin(test_days).values)
    if len(te_idx) == 0 or len(tr_idx) == 0:
        raise RuntimeError("time split produced empty train or test set (real failure).")
    return tr_idx, te_idx


def _group_split(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, *, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, te_idx


def _precision_at_k_by_day(df_eval: pd.DataFrame, proba: np.ndarray, *, k: int) -> Dict[str, float]:
    if "day" not in df_eval.columns:
        return {"precision_at_k": float("nan"), "lift_at_k": float("nan")}
    tmp = df_eval[["day", "label"]].copy()
    tmp["proba"] = proba
    tmp["day"] = tmp["day"].astype(str)
    base_rate = float(tmp["label"].mean()) if len(tmp) else float("nan")
    precs = []
    for _, g in tmp.groupby("day", sort=False):
        gg = g.sort_values("proba", ascending=False).head(int(k))
        if len(gg) == 0:
            continue
        precs.append(float(gg["label"].mean()))
    precision = float(np.mean(precs)) if precs else float("nan")
    lift = (precision / base_rate) if (base_rate and np.isfinite(base_rate) and base_rate > 0) else float("nan")
    return {"precision_at_k": precision, "lift_at_k": lift, "base_rate": base_rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["liquid", "outlier"], required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--split", choices=["time", "group"], default="time")
    ap.add_argument("--model", choices=["hgb", "gbdt"], default="hgb")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    df = _load_ds(args.ds_dir)

    # Required columns
    if "label" not in df.columns or "symbol" not in df.columns:
        raise RuntimeError("Dataset missing required columns label/symbol")
    for f in FEATURES:
        if f not in df.columns:
            raise RuntimeError(f"Missing feature {f} in dataset columns={list(df.columns)}")

    # Mode filter: real, deterministic
    if args.mode == "outlier":
        df = df[(df["mom5"].abs() >= 0.01) | (df["mom20"].abs() >= 0.02) | (df["gap_pct"].abs() >= 0.01)]
    if df.empty:
        raise RuntimeError("No rows left after mode filter (real failure).")

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["label"].astype(int).values
    groups = df["symbol"].astype(str).values

    if args.split == "time":
        tr_idx, te_idx = _time_split(df, test_size=float(args.test_size))
    else:
        tr_idx, te_idx = _group_split(X, y, groups, test_size=float(args.test_size))

    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    if len(set(y_test)) < 2:
        raise RuntimeError("Test set collapsed to a single class (real failure). Increase data range / adjust split.")

    if args.model == "hgb":
        model = HistGradientBoostingClassifier(
            random_state=42,
            learning_rate=0.08,
            max_depth=6,
            max_iter=250,
            l2_regularization=0.0,
        )
    else:
        model = GradientBoostingClassifier(random_state=42)

    model.fit(X_train, y_train)

    # Scores
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # HGB has predict_proba; this is just defensive
        proba = model.predict(X_test)

    auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))
    pk = _precision_at_k_by_day(df.iloc[te_idx], proba, k=int(args.topk))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": FEATURES,
        "mode": args.mode,
        "split": args.split,
        "model_kind": args.model,
        "auc": auc,
        "pr_auc": pr_auc,
        "topk": int(args.topk),
        "precision_at_k": float(pk.get("precision_at_k", float("nan"))),
        "lift_at_k": float(pk.get("lift_at_k", float("nan"))),
        "base_rate": float(pk.get("base_rate", float("nan"))),
        "pos_rate": float(np.mean(y)),
        "rows": int(len(df)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"ds_dir": str(Path(args.ds_dir))},
    }

    with out.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"AUC={auc:.4f} PR-AUC={pr_auc:.4f} base={artifact['base_rate']:.4f} P@{args.topk}={artifact['precision_at_k']:.4f} lift={artifact['lift_at_k']:.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
