from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def _list_shards(ds_dir: Path, start_date: str | None, end_date: str | None) -> List[Path]:
    shards = sorted(ds_dir.glob("entry_now_*.parquet"))
    if not start_date and not end_date:
        return shards

    def _date_from_name(p: Path) -> str:
        # entry_now_YYYY-MM-DD.parquet
        stem = p.stem
        return stem.split("entry_now_")[-1]

    out: List[Path] = []
    for p in shards:
        d = _date_from_name(p)
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_dir", default="data/entry_now_ds")
    ap.add_argument("--start_date", default="")
    ap.add_argument("--end_date", default="")
    ap.add_argument("--out", default="models/entry_now_30m.pkl")
    ap.add_argument("--failures", default="models/entry_now_parquet_failures.jsonl")
    ap.add_argument("--meta", default="models/entry_now_parquet_meta.json")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    ds_dir = Path(args.ds_dir)
    if not ds_dir.exists():
        raise SystemExit(f"ds_dir not found: {ds_dir}")

    shards = _list_shards(ds_dir, args.start_date or None, args.end_date or None)
    if not shards:
        raise SystemExit("No parquet shards found")

    failures: List[dict] = []
    frames = []

    t0 = time.time()
    for i, p in enumerate(shards, start=1):
        try:
            df = pd.read_parquet(p)
            if df is None or df.empty:
                continue
            frames.append(df)
        except Exception as e:
            failures.append({"stage": "read_parquet", "file": str(p), "error": f"{type(e).__name__}: {e}"})
        if i % 10 == 0:
            print(f"[{i}/{len(shards)}] loaded shards... elapsed={time.time()-t0:.1f}s")

    # write failures
    fail_p = Path(args.failures)
    fail_p.parent.mkdir(parents=True, exist_ok=True)
    with fail_p.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")

    if not frames:
        raise SystemExit("No data loaded from shards")

    df_all = pd.concat(frames, ignore_index=True)
    if "y" not in df_all.columns or "group" not in df_all.columns:
        raise SystemExit(f"Shard schema missing y/group columns; columns={list(df_all.columns)}")

    y = df_all["y"].astype(int)
    groups = df_all["group"].astype(str).values

    drop_cols = {"y", "group", "ticker", "date"}
    feature_cols = [c for c in df_all.columns if c not in drop_cols]
    X = df_all[feature_cols].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(-999.0)

    pos_rate = float(y.mean())
    print(f"Loaded {len(X):,} samples from {len(shards)} shard(s). Positive rate={pos_rate:.4f}")

    gss = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=6,
        max_iter=400,
        min_samples_leaf=40,
        l2_regularization=0.2,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = float("nan")
    print(f"AUC={auc:.4f}")

    bundle = {
        "kind": "entry_now_model_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "ds_dir": str(ds_dir),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "shards": len(shards),
        },
        # Keep portable primitives in metadata fields.
        "feature_names": list(map(str, feature_cols)),
        "pos_rate": float(pos_rate),
        "auc": float(auc),
        "model": model,
        "fillna_value": float(-999.0),
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # IMPORTANT: Use stdlib pickle (not joblib) to avoid non-portable wrappers
    # that can embed platform/version-specific objects (eg datetime fold state).
    with out_p.open("wb") as f:
        pickle.dump(bundle, f, protocol=4)
    print("Wrote", out_p)

    meta = {
        "samples": int(len(X)),
        "pos_rate": pos_rate,
        "auc": auc,
        "features": feature_cols,
        "failures": len(failures),
    }
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
