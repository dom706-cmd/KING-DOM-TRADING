from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import multiprocessing as mp
from dataclasses import asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ml.entry_now_dataset import (
    EntryNowParams,
    _find_col,
    _normalize_minute_flat,
    _slice_rth_1m,
    _slice_pm_1m,
    _opening_range_5m_from_1m,
    _today_dollar_vol_so_far,
    _stop_from_structure,
    _hit_order_within_horizon,
)
from scanner.indicators import vwap as vwap_series, trend_state_1m


# --- Multiprocessing worker context ----------------------------------------
# macOS uses the "spawn" start method for multiprocessing by default, which
# requires the worker target to be pickleable (i.e., defined at module scope).
# We pass context into workers via a Pool initializer.
_G_ALLOWED: set[str] | None = None
_G_DAILY_MAP: Dict[Tuple[str, date], Tuple[float, float]] | None = None
_G_PARAMS: EntryNowParams | None = None
_G_SESSION: str | None = None
_G_OUT_DIR: Path | None = None


def _init_worker(
    allowed: set[str],
    daily_map: Dict[Tuple[str, date], Tuple[float, float]],
    p: EntryNowParams,
    session: str,
    out_dir: Path,
) -> None:
    global _G_ALLOWED, _G_DAILY_MAP, _G_PARAMS, _G_SESSION, _G_OUT_DIR
    _G_ALLOWED = allowed
    _G_DAILY_MAP = daily_map
    _G_PARAMS = p
    _G_SESSION = session
    _G_OUT_DIR = out_dir


def _worker_day(fp_s: str):
    """Process a single gzip day file and write its Parquet shard.

    Returns (fp_s, n_samples, n_positive, failures_list).
    """
    if _G_ALLOWED is None or _G_DAILY_MAP is None or _G_PARAMS is None or _G_OUT_DIR is None:
        raise RuntimeError("Worker not initialized; _init_worker was not called")
    local_fail: List[dict] = []
    n, pos = _build_samples_for_day(
        day_path=Path(fp_s),
        allowed=_G_ALLOWED,
        daily_map=_G_DAILY_MAP,
        p=_G_PARAMS,
        out_dir=_G_OUT_DIR,
        failures=local_fail,
    )
    return fp_s, n, pos, local_fail


def _is_common_equity(sym: str) -> bool:
    s = sym.strip().upper()
    if not s or "$" in s:
        return False
    if "." in s:
        return False
    return s.isascii()


def _load_symbols(universe: str, offset: int, max_symbols: int) -> List[str]:
    uni = (universe or "").strip().lower()
    if uni == "file":
        p = Path("symbols.txt")
        if not p.exists():
            raise SystemExit("symbols.txt not found for universe=file")
        syms = [
            ln.strip().upper()
            for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        from universe.nasdaq_symbols import get_nasdaq_symbols

        syms = get_nasdaq_symbols(cache_dir="cache", ttl_seconds=86400)
    syms = [s for s in syms if _is_common_equity(s)]
    return syms[int(offset) : int(offset) + int(max_symbols)]


def _iter_minute_files(flat_minute_root: Path, start: date, end: date) -> List[Path]:
    files: List[Path] = []
    d = start
    while d <= end:
        p = flat_minute_root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.csv.gz"
        if p.exists():
            files.append(p)
        d += timedelta(days=1)
    return files


def _load_daily_table(flat_daily_root: Path, start: date, end: date, failures: List[dict]) -> pd.DataFrame:
    """Load daily bars for [start-90d, end] from day_aggs_v1 flatfiles.

    We only need: ticker, date, Close, Volume.
    Supports timestamps as ISO date or epoch (s/ms/us/ns) via pandas.
    """
    # Expand lookback for rolling features
    start2 = start - timedelta(days=120)

    frames: List[pd.DataFrame] = []
    files = _iter_minute_files(flat_daily_root, start2, end)
    for f in files:
        try:
            df = pd.read_csv(f, compression="gzip")
            sym_col = _find_col(df, "symbol", "ticker", "sym", "S")
            ts_col = _find_col(df, "date", "day", "timestamp", "t", "window_start")
            c_col = _find_col(df, "close", "c")
            v_col = _find_col(df, "volume", "v")
            if sym_col is None or ts_col is None or c_col is None or v_col is None:
                raise RuntimeError(f"daily schema unexpected; columns={list(df.columns)}")

            out = df[[sym_col, ts_col, c_col, v_col]].copy()
            out.columns = ["ticker", "ts", "Close", "Volume"]

            ts = out["ts"]
            if np.issubdtype(ts.dtype, np.number):
                mx = float(ts.max())
                unit = "s"
                if mx > 1e18:
                    unit = "ns"
                elif mx > 1e15:
                    unit = "us"
                elif mx > 1e12:
                    unit = "ms"
                out["date"] = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True).dt.tz_convert("America/New_York").dt.date
            else:
                out["date"] = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date

            out["ticker"] = out["ticker"].astype(str).str.upper()
            out = out.drop(columns=["ts"]).dropna(subset=["Close", "Volume", "date"])
            frames.append(out)
        except Exception as e:
            failures.append({"stage": "daily_flat_read", "file": str(f), "error": f"{type(e).__name__}: {e}"})

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "Close", "Volume"])

    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["ticker", "date"])
    return daily


def _build_daily_feature_map(daily: pd.DataFrame) -> Dict[Tuple[str, date], Tuple[float, float]]:
    """Return map[(ticker, date)] -> (avg20_dollar_vol_prior, prev_close).

    avg20_dollar_vol_prior is based only on prior days (shifted).
    """
    m: Dict[Tuple[str, date], Tuple[float, float]] = {}
    if daily.empty:
        return m

    # rolling avg20 volume shifted by 1
    g = daily.groupby("ticker", sort=False)
    daily["avg20_vol_prior"] = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).mean().shift(1))
    daily["prev_close"] = g["Close"].shift(1)
    daily["avg20_dollar_vol_prior"] = daily["avg20_vol_prior"] * daily["prev_close"]

    for r in daily.itertuples(index=False):
        key = (str(r.ticker), r.date)
        av = float(r.avg20_dollar_vol_prior) if np.isfinite(getattr(r, "avg20_dollar_vol_prior", np.nan)) else np.nan
        pc = float(r.prev_close) if np.isfinite(getattr(r, "prev_close", np.nan)) else np.nan
        m[key] = (av, pc)
    return m


def _read_minute_day_file(path: Path, usecols: List[str] | None, failures: List[dict]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, compression="gzip", usecols=usecols)

        # Normalize Massive flatfile schema to internal expected columns.
        # Some flatfiles use: ticker, window_start
        # Internally we expect: symbol (or ticker) and ts-like column handled downstream.
        if "symbol" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        if "ts" not in df.columns:
            if "window_start" in df.columns:
                df["ts"] = df["window_start"]
            elif "timestamp" in df.columns:
                df["ts"] = df["timestamp"]
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"])

        return df
    except Exception as e:
        failures.append({"stage": "minute_flat_read", "file": str(path), "error": f"{type(e).__name__}: {e}"})
        return pd.DataFrame()


def _build_samples_for_day(
    *,
    day_path: Path,
    allowed: set[str],
    daily_map: Dict[Tuple[str, date], Tuple[float, float]],
    p: EntryNowParams,
    out_dir: Path,
    failures: List[dict],
) -> Tuple[int, int]:
    """Build and write one parquet shard for a single trading day file."""

    # Determine the trading date from filename
    try:
        d = date.fromisoformat(day_path.stem.split(".")[0])
    except Exception:
        failures.append({"stage": "day_parse", "file": str(day_path), "error": "could_not_parse_date"})
        return 0, 0

    out_path = out_dir / f"entry_now_{d.isoformat()}.parquet"
    if out_path.exists():
        return 0, 0

    # Read minimal columns
    df = _read_minute_day_file(day_path, usecols=None, failures=failures)
    if df.empty:
        return 0, 0

    sym_col = _find_col(df, "symbol", "ticker", "sym", "S")
    if sym_col is None:
        failures.append({"stage": "minute_schema", "file": str(day_path), "error": f"missing ticker column; cols={list(df.columns)}"})
        return 0, 0

    df[sym_col] = df[sym_col].astype(str).str.upper()
    if allowed:
        df = df[df[sym_col].isin(allowed)]
    if df.empty:
        return 0, 0

    # Normalize time/OHLCV (keeps UTC tz index)
    try:
        df_norm = _normalize_minute_flat(df)
    except Exception as e:
        failures.append({"stage": "minute_normalize", "file": str(day_path), "error": f"{type(e).__name__}: {e}"})
        return 0, 0

    # Need the symbol column after normalization; we kept only OHLCV. Reattach symbol for grouping.
    # We'll rebuild a minimal frame with symbol + index aligned.
    # _normalize_minute_flat drops symbol; so we re-read symbol + timestamp columns and align by index.
    ts_col = _find_col(df, "timestamp", "time", "t", "window_start", "start", "datetime")
    if ts_col is None:
        failures.append({"stage": "minute_schema", "file": str(day_path), "error": f"missing timestamp column; cols={list(df.columns)}"})
        return 0, 0

    sym_series = df[sym_col].astype(str).str.upper().reset_index(drop=True)
    ts_series = df[ts_col].reset_index(drop=True)

    # Build index in UTC matching normalization
    try:
        if np.issubdtype(ts_series.dtype, np.number):
            mx = float(ts_series.max())
            unit = "s"
            if mx > 1e18:
                unit = "ns"
            elif mx > 1e15:
                unit = "us"
            elif mx > 1e12:
                unit = "ms"
            idx = pd.to_datetime(ts_series.astype("int64"), unit=unit, utc=True)
        else:
            idx = pd.to_datetime(ts_series, utc=True, errors="raise")
    except Exception as e:
        failures.append({"stage": "ts_parse", "file": str(day_path), "error": f"{type(e).__name__}: {e}"})
        return 0, 0

    df2 = df_norm.copy()
    df2["ticker"] = sym_series.values
    df2.index = idx
    df2 = df2.sort_index()

    rows = []
    ys = []
    groups = []
    tickers = []
    dates = []

    horizon_bars = max(1, int(p.horizon_min))
    stride = max(1, int(p.sample_every_min))

    for ticker, g in df2.groupby("ticker", sort=False):
        if g.empty:
            continue

        # Slice to session for this ET day
        try:
            sess = _slice_pm_1m(g.drop(columns=["ticker"]), pd.Timestamp(datetime(d.year, d.month, d.day))) if (_G_SESSION == "pm") else _slice_sess_1m(g.drop(columns=["ticker"]), pd.Timestamp(datetime(d.year, d.month, d.day)))
        except Exception:
            continue
        if sess is None or sess.empty or len(sess) < 60:
            continue

        # Daily prior features
        avg20_dv, prev_close = daily_map.get((ticker, d), (np.nan, np.nan))

        # VWAP series
        try:
            vw = vwap_series(sess)
        except Exception:
            vw = None

        # Opening range
        try:
            or_high, or_low = _opening_range_5m_from_1m(sess)
        except Exception:
            or_high = or_low = None

        end_i_max = min(len(sess) - horizon_bars - 1, 120)
        for i in range(5, end_i_max, stride):
            entry = float(sess["Close"].astype(float).iloc[i])
            if not np.isfinite(entry) or entry <= 0:
                continue
            if entry < float(p.min_price) or entry > float(p.max_price):
                continue

            # trend state
            try:
                sub = sess.iloc[: i + 1]
                sub_vw = vw.iloc[: i + 1] if vw is not None else None
                tr = trend_state_1m(sub, vw=sub_vw, lookback=int(p.lookback_trend_min))
                vwap_last = tr.get("vwap_last")
                vwap_delta_pct = tr.get("vwap_delta_pct")
                tstate = tr.get("state")
                slope = tr.get("slope_pct_lookback")
            except Exception:
                vwap_last = None
                vwap_delta_pct = None
                tstate = "unknown"
                slope = None

            # dollar vol so far
            try:
                dv_so_far = _today_dollar_vol_so_far(sess, i)
            except Exception:
                dv_so_far = None

            # RVOL approx
            try:
                today_vol_so_far = float(sess["Volume"].astype(float).iloc[: i + 1].sum())
                # Approx avg20 vol from avg20 dollar vol and prev_close if present
                avg20_vol = float(avg20_dv / prev_close) if (np.isfinite(avg20_dv) and np.isfinite(prev_close) and prev_close > 0) else np.nan
                frac = max(1e-6, (i + 1) / float(len(sess)))
                rvol_now = (today_vol_so_far / max(1e-9, avg20_vol * frac)) if np.isfinite(avg20_vol) and avg20_vol > 0 else None
            except Exception:
                rvol_now = None

            # OR distances
            dist_orh_pct = dist_orl_pct = or_range_pct = None
            if or_high and or_low:
                mid = (or_high + or_low) / 2.0
                or_range_pct = ((or_high - or_low) / max(1e-9, mid)) * 100.0
                dist_orh_pct = ((entry - or_high) / max(1e-9, or_high)) * 100.0
                dist_orl_pct = ((entry - or_low) / max(1e-9, or_low)) * 100.0

            for side in ("long", "short"):
                stop = _stop_from_structure(
                    side=side,
                    entry=entry,
                    rth_1m=sess,
                    end_i=i,
                    vw_last=vwap_last if (vwap_last is not None) else (float(vw.iloc[i]) if vw is not None else None),
                    or_high=or_high,
                    or_low=or_low,
                    p=p,
                )
                if stop is None:
                    continue
                R = abs(entry - float(stop))
                if R <= 0:
                    continue
                target = entry + 2.0 * R if side == "long" else entry - 2.0 * R
                future = sess.iloc[i + 1 : i + 1 + horizon_bars]
                y = _hit_order_within_horizon(side=side, entry=entry, stop=float(stop), target=float(target), future_1m=future)

                rows.append({
                    "side_long": 1.0 if side == "long" else 0.0,
                    "entry": entry,
                    "stop_dist_pct": (R / entry) * 100.0,
                    "vwap_delta_pct": float(vwap_delta_pct) if vwap_delta_pct is not None else np.nan,
                    "trend_slope_pct": float(slope) if slope is not None else np.nan,
                    "trend_state_up": 1.0 if tstate in ("up", "reclaim_vwap") else 0.0,
                    "trend_state_down": 1.0 if tstate in ("down", "lost_vwap") else 0.0,
                    "trend_state_chop": 1.0 if tstate == "chop" else 0.0,
                    "or_range_pct": float(or_range_pct) if or_range_pct is not None else np.nan,
                    "dist_orh_pct": float(dist_orh_pct) if dist_orh_pct is not None else np.nan,
                    "dist_orl_pct": float(dist_orl_pct) if dist_orl_pct is not None else np.nan,
                    "today_dollar_vol_so_far": float(dv_so_far) if dv_so_far is not None else np.nan,
                    "avg20_dollar_vol": float(avg20_dv) if np.isfinite(avg20_dv) else np.nan,
                    "rvol_now": float(rvol_now) if rvol_now is not None else np.nan,
                    "minutes_since_open": float(i),
                })
                ys.append(int(y))
                groups.append(f"{ticker}:{d.isoformat()}")
                tickers.append(ticker)
                dates.append(d.isoformat())

    if not rows:
        return 0, 0

    out = pd.DataFrame(rows)
    out["y"] = np.array(ys, dtype=int)
    out["group"] = np.array(groups, dtype=str)
    out["ticker"] = np.array(tickers, dtype=str)
    out["date"] = np.array(dates, dtype=str)

    # Write parquet
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    except Exception as e:
        failures.append({"stage": "write_parquet", "file": str(out_path), "error": f"{type(e).__name__}: {e}"})
        return 0, 0

    return int(len(out)), int(out["y"].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nasdaq")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_symbols", type=int, default=10000)
    ap.add_argument("--price_min", type=float, default=1.0)
    ap.add_argument("--price_max", type=float, default=30.0)
    ap.add_argument("--session", choices=["rth", "pm"], default="rth")
    ap.add_argument("--sample_every_min", type=int, default=3)
    ap.add_argument("--horizon_min", type=int, default=30)
    ap.add_argument("--include_premarket", action="store_true")
    ap.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--flat_minute_root", required=True)
    ap.add_argument("--flat_daily_root", default="")
    ap.add_argument("--out_dir", default="data/entry_now_ds")
    ap.add_argument("--failures", default="data/entry_now_ds/failures.jsonl")
    ap.add_argument("--meta", default="data/entry_now_ds/meta.json")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (per-day shards)")
    args = ap.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("end_date must be >= start_date")

    flat_minute_root = Path(args.flat_minute_root).expanduser()
    if not flat_minute_root.exists():
        raise SystemExit(f"flat_minute_root not found: {flat_minute_root}")

    flat_daily_root = Path(args.flat_daily_root).expanduser() if args.flat_daily_root else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: List[dict] = []

    # Universe tickers filter
    syms = _load_symbols(args.universe, args.offset, args.max_symbols)
    allowed = set(syms)

    # Daily map (recommended)
    daily_map: Dict[Tuple[str, date], Tuple[float, float]] = {}
    if flat_daily_root:
        if not flat_daily_root.exists():
            raise SystemExit(f"flat_daily_root not found: {flat_daily_root}")
        daily_tbl = _load_daily_table(flat_daily_root, start, end, failures)
        daily_map = _build_daily_feature_map(daily_tbl)
    else:
        failures.append({"stage": "daily_flat", "error": "flat_daily_root not provided; avg20_dollar_vol and rvol_now will be NaN"})

    p = EntryNowParams(
        horizon_min=int(args.horizon_min),
        sample_every_min=int(args.sample_every_min),
        min_price=float(args.price_min),
        max_price=float(args.price_max),
        include_premarket=bool(args.include_premarket) or (args.session == "pm"),
    )

    # Iterate minute files
    minute_files = _iter_minute_files(flat_minute_root, start, end)
    if not minute_files:
        raise SystemExit("No minute_aggs files found for the requested date range")

    total_samples = 0
    total_pos = 0
    built_days = 0

    t0 = time.time()

    # NOTE: multiprocessing on macOS (spawn) requires the worker to be a top-level
    # function (pickleable). We provide context via a module-level initializer.

    workers = max(1, int(args.workers))
    if workers == 1:
        _init_worker(allowed, daily_map, p, args.session, out_dir)
        for idx, fp in enumerate(minute_files, start=1):
            _, n, pos, local_fail = _worker_day(str(fp))
            failures.extend(local_fail)
            if n > 0:
                built_days += 1
                total_samples += n
                total_pos += pos
            if idx % 5 == 0:
                elapsed = time.time() - t0
                print(f"[{idx}/{len(minute_files)} days] shards={built_days} samples={total_samples:,} pos={total_pos:,} failures={len(failures):,} elapsed={elapsed:.1f}s")
    else:
        # Parallel per-day processing. Each worker reads one gz day file and writes its shard.
        # Note: failures are aggregated from workers (real exceptions only).
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(allowed, daily_map, p, args.session, out_dir),
        ) as pool:
            for idx, (fp_s, n, pos, local_fail) in enumerate(pool.imap_unordered(_worker_day, [str(p) for p in minute_files]), start=1):
                failures.extend(local_fail)
                if n > 0:
                    built_days += 1
                    total_samples += n
                    total_pos += pos
                if idx % 5 == 0:
                    elapsed = time.time() - t0
                    print(f"[{idx}/{len(minute_files)} days] shards={built_days} samples={total_samples:,} pos={total_pos:,} failures={len(failures):,} elapsed={elapsed:.1f}s")

    # Write failures/meta
    fail_p = Path(args.failures)
    fail_p.parent.mkdir(parents=True, exist_ok=True)
    with fail_p.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")

    meta = {
        "kind": "entry_now_dataset_v1",
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "flat_minute_root": str(flat_minute_root),
        "flat_daily_root": str(flat_daily_root) if flat_daily_root else "",
        "universe": args.universe,
        "offset": int(args.offset),
        "max_symbols": int(args.max_symbols),
        "params": asdict(p),
        "built_days": built_days,
        "samples": total_samples,
        "pos": total_pos,
        "pos_rate": (float(total_pos) / float(total_samples)) if total_samples else None,
        "failures": len(failures),
    }
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {built_days} shard(s) to {out_dir}")
    print(f"Total samples={total_samples:,} pos={total_pos:,} pos_rate={(float(total_pos)/float(total_samples)) if total_samples else 0.0:.4f}")


if __name__ == "__main__":
    main()
