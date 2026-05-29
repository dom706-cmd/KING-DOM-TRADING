from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def _shard_day(p: Path) -> str:
    stem = p.stem  # orb_ranker_YYYY-MM-DD
    if "orb_ranker_" in stem:
        return stem.split("orb_ranker_")[-1]
    return stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", default="data/orb_ranker_ds_under30")
    ap.add_argument("--out", default="models/orb_ranker_under30.pkl")
    ap.add_argument("--failures", default="models/orb_ranker_parquet_failures.jsonl")
    ap.add_argument("--meta", default="models/orb_ranker_parquet_meta.json")
    args = ap.parse_args()

    ds_dir = Path(args.ds_dir)
    shards = sorted(ds_dir.glob("orb_ranker_*.parquet"))
    if not shards:
        raise SystemExit(f"No shards found in {ds_dir}")

    frames: List[pd.DataFrame] = []
    groups: List[str] = []
    for p in shards:
        df = pd.read_parquet(p)
        if df is None or df.empty:
            continue
        g = _shard_day(p)
        df["_group"] = g
        frames.append(df)
    if not frames:
        raise SystemExit("All shards empty")

    data = pd.concat(frames, ignore_index=True)

    if "y" not in data.columns:
        raise SystemExit("Missing y in dataset")

    y = data["y"].astype(int)
    groups = data["_group"].astype(str)

    drop_cols = {"y", "_group"}
    X = data[[c for c in data.columns if c not in drop_cols]].copy()

    # Keep only numeric feature columns for sklearn
    non_features = []
    for c in X.columns:
        if X[c].dtype == object:
            non_features.append(c)
    for c in non_features:
        X = X.drop(columns=[c])

    feature_names = list(X.columns)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    try:
        p = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, p))
    except Exception:
        auc = None

    bundle = {
        "kind": "orb_ranker_model_from_shards_v1",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(X)),
        "pos": int(y.sum()),
        "pos_rate": float(y.mean()),
        "auc": auc,
        "feature_names": feature_names,
        "fillna_value": -999.0,
        "model": model,
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_p)
    print(f"Wrote: {out_p} (rows={len(X):,}, pos_rate={y.mean():.4f}, auc={auc})")

    meta_p = Path(args.meta)
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p.write_text(json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
