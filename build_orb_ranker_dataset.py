from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import date, timedelta
from typing import List, Dict

import pandas as pd

from providers.alpaca_provider import AlpacaProvider
from train_ranker import _fetch_intraday_5m_multi_year, _features_day, _label_day


def _days_in_5m(df5: pd.DataFrame) -> list[pd.Timestamp]:
    if df5 is None or df5.empty:
        return []
    # df index is expected UTC timestamps
    dset = sorted({ts.tz_convert("America/New_York").date() for ts in df5.index})
    return [pd.Timestamp(d) for d in dset]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_file", default="symbols.txt", help="One symbol per line")
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--label_minutes", type=int, default=30)
    ap.add_argument("--chunk_days", type=int, default=30)
    ap.add_argument("--timeout_s", type=int, default=20)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max_price", type=float, default=30.0, help="Hard constraint: only keep OR entry < max_price")
    ap.add_argument("--out_dir", default="data/orb_ranker_ds_under30")
    ap.add_argument("--failures", default="data/orb_ranker_ds_under30/failures.jsonl")
    ap.add_argument("--meta", default="data/orb_ranker_ds_under30/meta.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_path = Path(args.failures)
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    failures: List[dict] = []

    syms: list[str] = []
    sf = Path(args.symbols_file)
    if not sf.exists():
        raise SystemExit(f"symbols_file not found: {sf}")
    for line in sf.read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            syms.append(s)
    if not syms:
        raise SystemExit("symbols_file is empty")

    provider = AlpacaProvider()

    total_rows = 0
    total_pos = 0
    shards_written = 0

    for sym in syms:
        try:
            df5 = _fetch_intraday_5m_multi_year(
                provider,
                sym,
                years=float(args.years),
                chunk_days=int(args.chunk_days),
                timeout_s=int(args.timeout_s),
                retries=int(args.retries),
                failures=failures,
            ).sort_index()
            if df5 is None or df5.empty:
                continue

            daily = provider.get_daily_history(sym, period=f"{max(120, int(args.years*260)+60)}d").sort_index()
            if daily is None or daily.empty:
                failures.append({"symbol": sym, "stage": "daily", "error": "daily_empty"})
                continue

            for day in _days_in_5m(df5):
                # shard per day: entry is OR high (first RTH 5m bar high)
                try:
                    # Compute label first so invalid OR risk raises and gets recorded
                    y = int(_label_day(df5, day, label_minutes=int(args.label_minutes)))

                    feats = _features_day(daily, df5, day)

                    # Under-$max_price filter using OR-high entry
                    day_df = df5.tz_convert("America/New_York")
                    d0 = pd.Timestamp(day).tz_localize("America/New_York")
                    rth = day_df[(day_df.index >= d0.normalize() + pd.Timedelta(hours=9, minutes=30)) &
                                 (day_df.index < d0.normalize() + pd.Timedelta(hours=16, minutes=0))]
                    if rth is None or rth.empty:
                        raise RuntimeError("No regular-session bars for day")
                    entry = float(rth.iloc[0]["High"])
                    if not (entry > 0):
                        raise RuntimeError("Invalid OR entry")
                    if float(entry) >= float(args.max_price):
                        continue

                    row = {"symbol": sym, "day": str(day.date()), "y": y, "entry": entry}
                    row.update(feats)

                    out_p = out_dir / f"orb_ranker_{day.date().isoformat()}.parquet"
                    pd.DataFrame([row]).to_parquet(out_p, index=False)
                    shards_written += 1
                    total_rows += 1
                    total_pos += int(y)
                except Exception as e:
                    failures.append({"symbol": sym, "stage": "day", "day": str(day.date()), "error": f"{type(e).__name__}: {e}"})
        except Exception as e:
            failures.append({"symbol": sym, "stage": "symbol", "error": f"{type(e).__name__}: {e}"})

    # write failures
    with open(fail_path, "w", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec) + "\n")

    meta = {
        "kind": "orb_ranker_dataset_v1",
        "out_dir": str(out_dir),
        "symbols": len(syms),
        "years": float(args.years),
        "label_minutes": int(args.label_minutes),
        "max_price": float(args.max_price),
        "shards_written": int(shards_written),
        "total_rows": int(total_rows),
        "pos": int(total_pos),
        "pos_rate": (float(total_pos) / float(total_rows)) if total_rows else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
