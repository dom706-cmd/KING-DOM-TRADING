#!/usr/bin/env python3
# build_ab_dataset_from_flatfiles.py
#
# Build day-wise parquet shards for A/B (liquid/outlier) models using ONLY local Massive flatfiles.
#
# - Features: 5 daily-only features used by existing model_a/model_b bundles:
#     trend_20_50, vol20, avg20_dollar_vol, mom5, mom20
#   computed from daily flatfiles with a 1-day lag (no lookahead leakage).
#
# - Label: ORB-style 2R-in-window, computed from the day's intraday 1m flatfile
#   resampled to 5m bars (first 5m bar is OR). Label window uses label_minutes.
#
# No fake/sample data. Missing files are recorded as real failures.

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        return pa, pq
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to write parquet output. Install dependencies from requirements.txt "
            "before running this script."
        ) from e



def _to_utc_datetime_series(x: pd.Series) -> pd.Series:
    """Parse Massive window_start / timestamp to UTC datetime (robust)."""
    if x is None:
        return pd.to_datetime(x, utc=True, errors="coerce")
    if np.issubdtype(x.dtype, np.datetime64):
        return pd.to_datetime(x, utc=True, errors="coerce")
    if np.issubdtype(x.dtype, np.number):
        v = pd.to_numeric(x, errors="coerce")
        mx = float(np.nanmax(v.values)) if len(v) else float("nan")
        if not np.isfinite(mx):
            return pd.to_datetime(v, utc=True, errors="coerce")
        if mx > 1e18:
            unit = "ns"
        elif mx > 1e15:
            unit = "us"
        elif mx > 1e12:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(v, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(x, utc=True, errors="coerce")



def _find_col(df: pd.DataFrame, *cands: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _normalize_minute_flat(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    ts_col = _find_col(df, "timestamp", "time", "t", "window_start", "start", "datetime")
    sym_col = _find_col(df, "symbol", "ticker", "sym", "S")
    o_col = _find_col(df, "open", "o", "Open")
    h_col = _find_col(df, "high", "h", "High")
    l_col = _find_col(df, "low", "l", "Low")
    c_col = _find_col(df, "close", "c", "Close")
    v_col = _find_col(df, "volume", "v", "Volume")

    missing = [("timestamp", ts_col), ("symbol", sym_col), ("open", o_col), ("high", h_col),
               ("low", l_col), ("close", c_col), ("volume", v_col)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise RuntimeError(f"flatfile minute schema missing {missing}; columns={list(df.columns)}")

    out = df[[ts_col, sym_col, o_col, h_col, l_col, c_col, v_col]].copy()
    out.columns = ["ts", "symbol", "Open", "High", "Low", "Close", "Volume"]

    ts = out["ts"]
    if np.issubdtype(ts.dtype, np.number):
        mx = float(ts.max())
        if mx > 1e18:
            unit = "ns"
        elif mx > 1e15:
            unit = "us"
        elif mx > 1e12:
            unit = "ms"
        else:
            unit = "s"
        out.index = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True)
    else:
        out.index = pd.to_datetime(ts, utc=True, errors="raise")
    out = out.drop(columns=["ts"])
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out.sort_index()



def _normalize_daily_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Massive day_aggs_v1 (or similar) into a consistent schema.

    We accept various column spellings and derive `day` from either a date/day column
    or from `window_start` / timestamp-like columns.

    Output columns:
      symbol (upper), day (python date), Open, High, Low, Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    day_col = _find_col(df, "date", "day")
    ws_col = _find_col(df, "window_start", "timestamp", "time", "t")
    sym_col = _find_col(df, "symbol", "ticker", "sym", "S")

    o_col = _find_col(df, "open", "o", "Open")
    h_col = _find_col(df, "high", "h", "High")
    l_col = _find_col(df, "low", "l", "Low")
    c_col = _find_col(df, "close", "c", "Close")
    v_col = _find_col(df, "volume", "v", "Volume")

    missing = []
    if sym_col is None:
        missing.append("symbol/ticker")
    if c_col is None:
        missing.append("close")
    if v_col is None:
        missing.append("volume")
    if day_col is None and ws_col is None:
        missing.append("date/day or window_start")
    if missing:
        raise RuntimeError(f"flatfile daily schema missing {missing}; columns={list(df.columns)}")

    # Parse day
    if day_col is not None:
        day_raw = df[day_col]
        dt = _to_utc_datetime_series(day_raw)
    else:
        day_raw = df[ws_col]
        dt = _to_utc_datetime_series(day_raw)
    if dt.isna().all():
        raise RuntimeError(
            f"flatfile daily day/window_start could not be parsed; sample={day_raw.head(5).tolist()}"
        )

    out = pd.DataFrame({
        "day": dt.dt.date,
        "symbol": df[sym_col].astype(str).str.upper(),
        "Close": pd.to_numeric(df[c_col], errors="coerce"),
        "Volume": pd.to_numeric(df[v_col], errors="coerce"),
    })

    # Open/High/Low are optional but strongly preferred for gold-standard features.
    if o_col is not None:
        out["Open"] = pd.to_numeric(df[o_col], errors="coerce")
    else:
        out["Open"] = np.nan
    if h_col is not None:
        out["High"] = pd.to_numeric(df[h_col], errors="coerce")
    else:
        out["High"] = np.nan
    if l_col is not None:
        out["Low"] = pd.to_numeric(df[l_col], errors="coerce")
    else:
        out["Low"] = np.nan

    out = out.dropna(subset=["symbol", "day", "Close", "Volume"])
    return out


def _day_paths(root: Path, start: date, end: date) -> List[Path]:
    paths: List[Path] = []
    d = start
    while d <= end:
        p = root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.csv.gz"
        if p.exists():
            paths.append(p)
        d += timedelta(days=1)
    return paths


def _resample_1m_to_5m_rth(df_1m: pd.DataFrame, day: date) -> pd.DataFrame:
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    idx_et = df_1m.index.tz_convert(_ET)
    df = df_1m.copy()
    df["__ts_et"] = idx_et
    d0 = datetime(day.year, day.month, day.day, tzinfo=_ET)
    rth_start = d0.replace(hour=9, minute=30)
    rth_end = d0.replace(hour=16, minute=0)
    df = df[(df["__ts_et"] >= rth_start) & (df["__ts_et"] < rth_end)]
    if df.empty:
        return pd.DataFrame()
    df = df.drop(columns=["__ts_et"])

    df_et = df.copy()
    df_et.index = idx_et[(idx_et >= rth_start) & (idx_et < rth_end)]
    o = df_et["Open"].resample("5min", label="left", closed="left").first()
    h = df_et["High"].resample("5min", label="left", closed="left").max()
    l = df_et["Low"].resample("5min", label="left", closed="left").min()
    c = df_et["Close"].resample("5min", label="left", closed="left").last()
    v = df_et["Volume"].resample("5min", label="left", closed="left").sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = out.dropna()
    return out


def _label_orb_2r(df5_rth: pd.DataFrame, *, label_minutes: int) -> int:
    if df5_rth is None or df5_rth.empty:
        return 0
    df = df5_rth.sort_index()
    if len(df) < 2:
        return 0
    first = df.iloc[0]
    entry = float(first["High"])
    stop = float(first["Low"])
    risk = entry - stop
    if not (risk > 0 and math.isfinite(risk)):
        return 0
    target = entry + 2.0 * risk
    max_bars = max(1, int(label_minutes) // 5)
    ahead = df.iloc[1: 1 + max_bars]
    in_pos = False
    for _, row in ahead.iterrows():
        hi = float(row["High"]); lo = float(row["Low"])
        if not in_pos:
            if hi >= entry:
                in_pos = True
            else:
                continue
        if lo <= stop:
            return 0
        if hi >= target:
            return 1
    return 0


def _compute_daily_features_panel(daily_paths: List[Path]) -> pd.DataFrame:
    """Build a per-(symbol, day) feature panel from daily bars.

    Gold-standard constraints:
      - No lookahead leakage for rolling stats: all rolling features are shifted by 1 day.
      - We may use *today's Open* to compute gap% (known at RTH open).
      - We do NOT use today's High/Low/Close for predictive features unless shifted.

    Adds these features:
      trend_20_50, vol20, avg20_dollar_vol, mom5, mom20
      gap_pct, atr14_pct, avg20_vol
    """
    frames: List[pd.DataFrame] = []
    for p in daily_paths:
        try:
            df = pd.read_csv(p, compression="gzip", low_memory=False)
        except ValueError:
            # Some flatfiles have mixed numeric formatting; fall back to python engine.
            df = pd.read_csv(p, compression="gzip", low_memory=False, engine="python")
        d = _normalize_daily_flat(df)
        if d.empty:
            continue
        frames.append(d)

    if not frames:
        return pd.DataFrame()

    all_daily = pd.concat(frames, ignore_index=True).sort_values(["symbol", "day"])
    g = all_daily.groupby("symbol", sort=False)

    close = g["Close"]
    vol = g["Volume"]

    # Rolling trend signal (lagged)
    sma20 = close.rolling(20).mean().shift(1).reset_index(level=0, drop=True)
    sma50 = close.rolling(50).mean().shift(1).reset_index(level=0, drop=True)
    trend_20_50 = (sma20 > sma50).astype(float)

    # Volatility of daily returns (lagged)
    ret = close.pct_change()
    vol20 = ret.rolling(20).std().shift(1).reset_index(level=0, drop=True)

    # Average dollar volume (lagged)
    avg20_vol = vol.rolling(20).mean().shift(1).reset_index(level=0, drop=True)
    prev_close = close.shift(1)
    avg20_dollar_vol = (avg20_vol * prev_close).reset_index(level=0, drop=True)

    # Momentum (lagged)
    mom5 = (close.shift(1) / close.shift(6) - 1.0).replace([np.inf, -np.inf], np.nan).reset_index(level=0, drop=True)
    mom20 = (close.shift(1) / close.shift(21) - 1.0).replace([np.inf, -np.inf], np.nan).reset_index(level=0, drop=True)

    # Gap% (uses today's open, known at market open; uses prev close)
    if "Open" in all_daily.columns and "Close" in all_daily.columns:
        open_s = pd.to_numeric(all_daily["Open"], errors="coerce")
        prev_close_s = pd.to_numeric(all_daily["Close"], errors="coerce").groupby(all_daily["symbol"], sort=False).shift(1)
        gap_pct = ((open_s / prev_close_s) - 1.0).replace([np.inf, -np.inf], np.nan)
    else:
        gap_pct = pd.Series(np.nan, index=all_daily.index)

    # ATR14% (true range uses today's high/low and prev close; ATR is lagged by 1)
    if "High" in all_daily.columns and "Low" in all_daily.columns and "Close" in all_daily.columns:
        high_s = pd.to_numeric(all_daily["High"], errors="coerce")
        low_s = pd.to_numeric(all_daily["Low"], errors="coerce")
        close_s = pd.to_numeric(all_daily["Close"], errors="coerce")
        prev_close_s = close_s.groupby(all_daily["symbol"], sort=False).shift(1)

        tr1 = (high_s - low_s).abs()
        tr2 = (high_s - prev_close_s).abs()
        tr3 = (low_s - prev_close_s).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr14_raw = tr.groupby(all_daily["symbol"], sort=False).rolling(14).mean().reset_index(level=0, drop=True)
        atr14 = atr14_raw.groupby(all_daily["symbol"], sort=False).shift(1)
        atr14_pct = (atr14 / prev_close_s).replace([np.inf, -np.inf], np.nan)
    else:
        atr14_pct = pd.Series(np.nan, index=all_daily.index)

    out = all_daily.copy()
    out["trend_20_50"] = trend_20_50
    out["vol20"] = vol20
    out["avg20_vol"] = avg20_vol
    out["avg20_dollar_vol"] = avg20_dollar_vol
    out["mom5"] = mom5
    out["mom20"] = mom20
    out["gap_pct"] = gap_pct
    out["atr14_pct"] = atr14_pct
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat_minute_root", required=True)
    ap.add_argument("--flat_daily_root", required=True)
    ap.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--failures", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--label_minutes", type=int, default=60)
    ap.add_argument("--price_max", type=float, default=30.0)
    args = ap.parse_args()

    t0 = time.time()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    flat_min_root = Path(args.flat_minute_root).expanduser()
    flat_day_root = Path(args.flat_daily_root).expanduser()
    if not flat_min_root.exists():
        raise FileNotFoundError(f"flat_minute_root not found: {flat_min_root}")
    if not flat_day_root.exists():
        raise FileNotFoundError(f"flat_daily_root not found: {flat_day_root}")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    failures_path = Path(args.failures).expanduser() if args.failures else (out_dir / "failures.jsonl")
    meta_path = Path(args.meta).expanduser() if args.meta else (out_dir / "meta.json")

    lookback_start = start - timedelta(days=200)
    daily_paths = _day_paths(flat_day_root, lookback_start, end)
    if not daily_paths:
        raise FileNotFoundError(f"No daily flatfiles found in {lookback_start}..{end} under {flat_day_root}")
    daily_panel = _compute_daily_features_panel(daily_paths)
    if daily_panel.empty:
        raise RuntimeError("Daily panel empty after reading flatfiles")

    daily_panel.index = pd.MultiIndex.from_frame(daily_panel[["symbol", "day"]])
    daily_panel = daily_panel.sort_index()

    minute_paths = _day_paths(flat_min_root, start, end)
    if not minute_paths:
        raise FileNotFoundError(f"No minute flatfiles found in {start}..{end} under {flat_min_root}")

    total_days = shards = samples = pos = failures = 0

    with failures_path.open("w", encoding="utf-8") as ff:
        for p in minute_paths:
            day_str = p.name.replace(".csv.gz", "")
            day = date.fromisoformat(day_str)
            total_days += 1
            try:
                try:
                    raw = pd.read_csv(p, compression="gzip", low_memory=False)
                except ValueError:
                    # Some flatfiles have mixed numeric formatting; fall back to python engine.
                    raw = pd.read_csv(p, compression="gzip", low_memory=False, engine="python")

                m = _normalize_minute_flat(raw)
                if m.empty:
                    raise RuntimeError("minute file parsed empty")

                out_rows = []
                for sym, gdf in m.groupby("symbol"):
                    try:
                        feats = daily_panel.loc[(sym, day)]
                    except KeyError:
                        continue

                    # If duplicates exist for (symbol, day), take the first row deterministically.
                    if isinstance(feats, pd.DataFrame):
                        feats = feats.iloc[0]

                    close_today = float(feats.get("Close", np.nan))
                    if math.isfinite(close_today) and close_today > float(args.price_max):
                        continue

                    df5 = _resample_1m_to_5m_rth(gdf.drop(columns=["symbol"]), day)
                    if df5.empty:
                        continue

                    y = _label_orb_2r(df5, label_minutes=int(args.label_minutes))
                    
                    # Intraday features from the opening range (RTH) using 5m bars
                    first5 = df5.iloc[0]
                    or_open = float(first5["Open"])
                    or_high = float(first5["High"])
                    or_low = float(first5["Low"])
                    or_range_pct = (or_high - or_low) / (or_open if (or_open and math.isfinite(or_open)) else 1.0)

                    vol5 = float(first5["Volume"])
                    vol15 = float(df5.iloc[: min(3, len(df5))]["Volume"].sum())
                    avg20_vol = float(feats.get("avg20_vol", np.nan))
                    relvol5 = (vol5 / avg20_vol) if (math.isfinite(avg20_vol) and avg20_vol > 0) else float("nan")
                    relvol15 = (vol15 / avg20_vol) if (math.isfinite(avg20_vol) and avg20_vol > 0) else float("nan")

                    out_rows.append({
                        "symbol": sym,
                        "day": day.isoformat(),
                        "label": int(y),

                        # Daily features (lagged unless explicitly safe)
                        "trend_20_50": float(feats.get("trend_20_50", 0.0) or 0.0),
                        "vol20": float(feats.get("vol20", 0.0) or 0.0),
                        "avg20_dollar_vol": float(feats.get("avg20_dollar_vol", 0.0) or 0.0),
                        "mom5": float(feats.get("mom5", 0.0) or 0.0),
                        "mom20": float(feats.get("mom20", 0.0) or 0.0),
                        "gap_pct": float(feats.get("gap_pct", 0.0) or 0.0),
                        "atr14_pct": float(feats.get("atr14_pct", 0.0) or 0.0),

                        # Intraday opening-range features
                        "or_range_pct": float(or_range_pct),
                        "relvol5": float(relvol5) if math.isfinite(relvol5) else 0.0,
                        "relvol15": float(relvol15) if math.isfinite(relvol15) else 0.0,
                    })
                if out_rows:
                    day_dir = out_dir / f"day={day.isoformat()}"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    shard_path = day_dir / f"part-{day.isoformat()}.parquet"
                    pa, pq = _require_pyarrow()
                    table = pa.Table.from_pandas(pd.DataFrame(out_rows), preserve_index=False)
                    pq.write_table(table, shard_path)
                    shards += 1
                    samples += len(out_rows)
                    pos += int(sum(r["label"] for r in out_rows))
            except Exception as e:
                failures += 1
                ff.write(json.dumps({
                    "day": day_str,
                    "stage": "day_build",
                    "error": f"{type(e).__name__}: {e}",
                    "file": str(p),
                }) + "\n")

            if total_days % 5 == 0:
                elapsed = time.time() - t0
                print(f"[{total_days} days] shards={shards} samples={samples:,} pos={pos:,} failures={failures} elapsed={elapsed:.1f}s", flush=True)

    meta = {
        "kind": "ab_dataset_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "label_minutes": int(args.label_minutes),
        "price_max": float(args.price_max),
        "out_dir": str(out_dir),
        "flat_minute_root": str(flat_min_root),
        "flat_daily_root": str(flat_day_root),
        "feature_names": ["trend_20_50", "vol20", "avg20_dollar_vol", "mom5", "mom20", "gap_pct", "atr14_pct", "or_range_pct", "relvol5", "relvol15"],
        "shards": int(shards),
        "samples": int(samples),
        "pos": int(pos),
        "pos_rate": float(pos / max(1, samples)),
        "failures": int(failures),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {shards} shard(s) to {out_dir}")
    print(f"Total samples={samples:,} pos={pos:,} pos_rate={meta['pos_rate']:.4f}")


if __name__ == "__main__":
    main()
