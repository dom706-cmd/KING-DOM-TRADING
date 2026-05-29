from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from providers.alpaca_provider import AlpacaProvider
from ml.entry_now_dataset import EntryNowParams, build_entry_now_samples_for_symbol


def _is_common_equity(sym: str) -> bool:
    s = sym.strip().upper()
    if not s or "$" in s:
        return False
    if "." in s:
        # filter out common warrant/right suffixes; keep plain class shares like BRK.B? (we drop to stay safe)
        return False
    return s.isascii()


def _load_symbols(universe: str, offset: int, max_symbols: int) -> List[str]:
    uni = (universe or "").strip().lower()
    if uni == "file":
        p = Path("symbols.txt")
        if not p.exists():
            raise SystemExit("symbols.txt not found for universe=file")
        syms = [ln.strip().upper() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    else:
        from universe.nasdaq_symbols import get_nasdaq_symbols
        syms = get_nasdaq_symbols(cache_dir="cache", ttl_seconds=86400)
    syms = [s for s in syms if _is_common_equity(s)]
    return syms[int(offset): int(offset) + int(max_symbols)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nasdaq")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_symbols", type=int, default=1000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--sample_every_min", type=int, default=3)
    ap.add_argument("--horizon_min", type=int, default=30)
    ap.add_argument("--price_min", type=float, default=1.0)
    ap.add_argument("--price_max", type=float, default=30.0)
    ap.add_argument("--include_premarket", action="store_true")
    ap.add_argument("--flat_minute_root", default="")
    ap.add_argument("--flat_daily_root", default="")
    ap.add_argument("--out", default="models/entry_now_30m.pkl")
    ap.add_argument("--failures", default="models/entry_now_failures.jsonl")
    ap.add_argument("--attempts", type=int, default=2)
    args = ap.parse_args()

    provider = AlpacaProvider()
    syms = _load_symbols(args.universe, args.offset, args.max_symbols)
    if not syms:
        raise SystemExit("0 symbols loaded")

    p = EntryNowParams(
        horizon_min=int(args.horizon_min),
        sample_every_min=int(args.sample_every_min),
        min_price=float(args.price_min),
        max_price=float(args.price_max),
        include_premarket=bool(args.include_premarket),
    )

    failures: List[dict] = []
    Xs = []
    ys = []
    groups_all = []

    t0 = time.time()
    for idx, sym in enumerate(syms, start=1):
        # retries around build call (record real failures)
        last_err = None
        for attempt in range(1, max(1, int(args.attempts)) + 1):
            try:
                X, y, groups = build_entry_now_samples_for_symbol(
                    provider=provider,
                    symbol=sym,
                    p=p,
                    days_back=int(args.days),
                    failures=failures,
                    flat_minute_root=(args.flat_minute_root or None),
                    flat_daily_root=(args.flat_daily_root or None),
                )
                if not X.empty:
                    Xs.append(X)
                    ys.append(y)
                    groups_all.extend(groups)
                break
            except Exception as e:
                last_err = e
                failures.append({"symbol": sym, "stage": "build", "attempt": attempt, "error": f"{type(e).__name__}: {e}"})
        if idx % 50 == 0:
            elapsed = time.time() - t0
            n = int(sum(len(x) for x in Xs)) if Xs else 0
            pos = int(sum(int(v) for y in ys for v in y.values)) if ys else 0
            print(f"[{idx}/{len(syms)}] samples={n:,} pos={pos:,} failures={len(failures):,} elapsed={elapsed:.1f}s")

    # write failures (real)
    fail_p = Path(args.failures)
    fail_p.parent.mkdir(parents=True, exist_ok=True)
    with fail_p.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")

    if not Xs:
        raise SystemExit("No training samples built (all failures / filters)")

    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    groups = np.array(groups_all)

    # Fill missing values (real: missing is missing; we encode explicitly)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(-999.0)

    pos_rate = float(y.mean())
    print(f"Built {len(X):,} samples. Positive rate={pos_rate:.4f}")

    # Split by group to avoid leaking same symbol-day into both train/test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
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
        "params": asdict(p),
        "feature_names": list(X.columns),
        "pos_rate": pos_rate,
        "auc": auc,
        "model": model,
        "fillna_value": -999.0,
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_p)
    print("Wrote", out_p)


if __name__ == "__main__":
    main()
