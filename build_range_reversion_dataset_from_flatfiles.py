#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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



def _find_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _to_utc_datetime_series(x: pd.Series) -> pd.Series:
    if x is None:
        return pd.to_datetime(x, utc=True, errors="coerce")
    if np.issubdtype(x.dtype, np.number):
        mx = float(pd.to_numeric(x, errors="coerce").max())
        if mx > 1e18:
            unit = "ns"
        elif mx > 1e15:
            unit = "us"
        elif mx > 1e12:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(x.astype("int64"), unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(x, utc=True, errors="coerce")


def _day_paths(root: Path, start: date, end: date) -> List[Path]:
    paths: List[Path] = []
    d = start
    while d <= end:
        p = root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.csv.gz"
        if p.exists():
            paths.append(p)
        d += timedelta(days=1)
    return paths


def _normalize_daily_flat(df: pd.DataFrame) -> pd.DataFrame:
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

    if day_col is not None:
        dt = _to_utc_datetime_series(df[day_col])
    else:
        dt = _to_utc_datetime_series(df[ws_col])
    if dt.isna().all():
        raise RuntimeError("flatfile daily day/window_start could not be parsed")

    out = pd.DataFrame({
        "day": dt.dt.date,
        "symbol": df[sym_col].astype(str).str.upper(),
        "Close": pd.to_numeric(df[c_col], errors="coerce"),
        "Volume": pd.to_numeric(df[v_col], errors="coerce"),
    })
    out["Open"] = pd.to_numeric(df[o_col], errors="coerce") if o_col else np.nan
    out["High"] = pd.to_numeric(df[h_col], errors="coerce") if h_col else np.nan
    out["Low"]  = pd.to_numeric(df[l_col], errors="coerce") if l_col else np.nan
    out = out.dropna(subset=["symbol", "day", "Close", "Volume"])
    return out


def _compute_daily_features_panel(daily_paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in daily_paths:
        try:
            df = pd.read_csv(p, compression="gzip", low_memory=False)
        except ValueError:
            df = pd.read_csv(p, compression="gzip", low_memory=False, engine="python")
        d = _normalize_daily_flat(df)
        if not d.empty:
            frames.append(d)

    if not frames:
        return pd.DataFrame()

    all_daily = pd.concat(frames, ignore_index=True).sort_values(["symbol", "day"])
    g = all_daily.groupby("symbol", sort=False)

    close = g["Close"]
    vol = g["Volume"]

    sma20 = close.rolling(20).mean().shift(1).reset_index(level=0, drop=True)
    sma50 = close.rolling(50).mean().shift(1).reset_index(level=0, drop=True)
    trend_20_50 = (sma20 > sma50).astype(float)

    ret = close.pct_change()
    vol20 = ret.rolling(20).std().shift(1).reset_index(level=0, drop=True)

    avg20_vol = vol.rolling(20).mean().shift(1).reset_index(level=0, drop=True)
    prev_close = close.shift(1)
    avg20_dollar_vol = (avg20_vol * prev_close).reset_index(level=0, drop=True)

    mom5 = (close.shift(1) / close.shift(6) - 1.0).replace([np.inf, -np.inf], np.nan).reset_index(level=0, drop=True)
    mom20 = (close.shift(1) / close.shift(21) - 1.0).replace([np.inf, -np.inf], np.nan).reset_index(level=0, drop=True)

    # gap% uses today's Open (known at open) and prev close
    open_s = pd.to_numeric(all_daily["Open"], errors="coerce")
    prev_close_s = pd.to_numeric(all_daily["Close"], errors="coerce").groupby(all_daily["symbol"], sort=False).shift(1)
    gap_pct = ((open_s / prev_close_s) - 1.0).replace([np.inf, -np.inf], np.nan)

    # atr14% lagged by 1
    high_s = pd.to_numeric(all_daily["High"], errors="coerce")
    low_s = pd.to_numeric(all_daily["Low"], errors="coerce")
    close_s = pd.to_numeric(all_daily["Close"], errors="coerce")
    prev_close_s2 = close_s.groupby(all_daily["symbol"], sort=False).shift(1)
    tr = pd.concat([(high_s - low_s).abs(), (high_s - prev_close_s2).abs(), (low_s - prev_close_s2).abs()], axis=1).max(axis=1)
    atr14_raw = tr.groupby(all_daily["symbol"], sort=False).rolling(14).mean().reset_index(level=0, drop=True)
    atr14 = atr14_raw.groupby(all_daily["symbol"], sort=False).shift(1)
    atr14_pct = (atr14 / prev_close_s2).replace([np.inf, -np.inf], np.nan)

    out = all_daily.copy()
    out["trend_20_50"] = trend_20_50
    out["vol20"] = vol20
    out["avg20_dollar_vol"] = avg20_dollar_vol
    out["mom5"] = mom5
    out["mom20"] = mom20
    out["gap_pct"] = gap_pct
    out["atr14_pct"] = atr14_pct
    return out[["symbol","day","trend_20_50","vol20","avg20_dollar_vol","mom5","mom20","gap_pct","atr14_pct"]]


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
    tr_col = _find_col(df, "transactions", "n", "trade_count")

    missing = [("timestamp", ts_col), ("symbol", sym_col), ("open", o_col), ("high", h_col),
               ("low", l_col), ("close", c_col), ("volume", v_col)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise RuntimeError(f"flatfile minute schema missing {missing}; columns={list(df.columns)}")

    cols = [ts_col, sym_col, o_col, h_col, l_col, c_col, v_col]
    out = df[cols].copy()
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
        out["dt"] = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True)
    else:
        out["dt"] = pd.to_datetime(ts, utc=True, errors="raise")

    out = out.drop(columns=["ts"])
    out["symbol"] = out["symbol"].astype(str).str.upper()

    if tr_col:
        out["transactions"] = pd.to_numeric(df[tr_col], errors="coerce").fillna(0.0)
    else:
        out["transactions"] = 0.0

    return out


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    dt_et = df["dt"].dt.tz_convert(_ET)
    t = dt_et.dt.time
    return df[(t >= datetime.strptime("09:30","%H:%M").time()) & (t <= datetime.strptime("16:00","%H:%M").time())].copy()


def _rolling_slope(y: np.ndarray) -> float:
    if len(y) < 5:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = y.astype(float)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.polyfit(x, y, 1)[0])


def _label_success(after: pd.DataFrame, stop: float, vwap: float) -> int:
    stop_hits = (after["Low"].astype(float) <= stop).to_numpy()
    vwap_hits = (after["High"].astype(float) >= vwap).to_numpy()

    if stop_hits.any() and vwap_hits.any():
        stop_first = int(np.argmax(stop_hits))
        vwap_first = int(np.argmax(vwap_hits))
        return 1 if vwap_first < stop_first else 0
    return 1 if vwap_hits.any() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat_minute_root", required=True)
    ap.add_argument("--flat_daily_root", required=True)
    ap.add_argument("--start_date", required=True)
    ap.add_argument("--end_date", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--failures", default=None)

    ap.add_argument("--range_window_min", type=int, default=60)
    ap.add_argument("--band_k", type=float, default=2.0)
    ap.add_argument("--horizon_min", type=int, default=30)
    ap.add_argument("--cooldown_min", type=int, default=20)
    ap.add_argument("--stop_sigma_mult", type=float, default=0.75)
    ap.add_argument("--lookback_min", type=int, default=30)

    ap.add_argument("--max_symbols_per_day", type=int, default=1200)
    ap.add_argument("--min_price", type=float, default=0.5)
    ap.add_argument("--max_price", type=float, default=30.0)
    args = ap.parse_args()

    minute_root = Path(args.flat_minute_root)
    daily_root = Path(args.flat_daily_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures_path = Path(args.failures) if args.failures else (out_dir / "failures.jsonl")

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    # daily features need lookback for SMA50 etc
    daily_paths = _day_paths(daily_root, start - timedelta(days=70), end)
    daily_panel = _compute_daily_features_panel(daily_paths)
    if daily_panel.empty:
        raise RuntimeError("No daily data loaded; cannot build daily feature panel")

    daily_panel = daily_panel.dropna(subset=["day","symbol"])
    # Ensure unique (symbol, day); provider exports can contain duplicates
    dup_n = int(daily_panel.duplicated(subset=["symbol","day"]).sum())
    if dup_n:
        print(f"[WARN] daily_panel duplicate (symbol,day) rows={dup_n}; keeping last")
    daily_panel = daily_panel.drop_duplicates(subset=["symbol","day"], keep="last")

    daily_panel_key = daily_panel.set_index(["symbol","day"], drop=False)

    d = start
    total_events = 0
    while d <= end:
        minute_path = minute_root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.csv.gz"
        daily_path = daily_root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.csv.gz"

        if not minute_path.exists():
            with failures_path.open("a") as f:
                f.write(json.dumps({"day": d.isoformat(), "stage": "minute_file_missing", "path": str(minute_path)}) + "\n")
            d += timedelta(days=1)
            continue
        if not daily_path.exists():
            with failures_path.open("a") as f:
                f.write(json.dumps({"day": d.isoformat(), "stage": "daily_file_missing", "path": str(daily_path)}) + "\n")
            d += timedelta(days=1)
            continue

        # select universe for this day using daily (price + top dollar volume)
        try:
            day_daily_raw = pd.read_csv(daily_path, compression="gzip", low_memory=False)
        except ValueError:
            day_daily_raw = pd.read_csv(daily_path, compression="gzip", low_memory=False, engine="python")
        day_daily = _normalize_daily_flat(day_daily_raw)
        day_daily = day_daily[day_daily["day"] == d]
        day_daily["dollar_vol"] = day_daily["Close"] * day_daily["Volume"]
        day_daily = day_daily[(day_daily["Close"] >= args.min_price) & (day_daily["Close"] <= args.max_price)]
        day_daily = day_daily.sort_values("dollar_vol", ascending=False).head(args.max_symbols_per_day)
        symbols = set(day_daily["symbol"].astype(str).tolist())
        if not symbols:
            d += timedelta(days=1)
            continue

        # read minute file (can be big, so filter early after load)
        try:
            raw = pd.read_csv(minute_path, compression="gzip", low_memory=False)
        except ValueError:
            raw = pd.read_csv(minute_path, compression="gzip", low_memory=False, engine="python")
        m = _normalize_minute_flat(raw)
        m = m[m["symbol"].isin(symbols)]
        if m.empty:
            with failures_path.open("a") as f:
                f.write(json.dumps({"day": d.isoformat(), "stage": "minute_filtered_empty", "path": str(minute_path)}) + "\n")
            d += timedelta(days=1)
            continue
        m = _rth_only(m)
        if m.empty:
            d += timedelta(days=1)
            continue

        rows: List[Dict[str, float]] = []
        for sym, g in m.groupby("symbol", sort=False):
            g = g.sort_values("dt").reset_index(drop=True)
            if len(g) < max(120, args.range_window_min):
                continue

            tp = (g["High"] + g["Low"] + g["Close"]) / 3.0
            vol = g["Volume"].astype(float)

            minp = max(10, args.range_window_min // 3)
            w_num = (tp * vol).rolling(args.range_window_min, min_periods=minp).sum()
            w_den = vol.rolling(args.range_window_min, min_periods=minp).sum()
            vwap = (w_num / (w_den + 1e-12)).astype(float)
            sigma = g["Close"].rolling(args.range_window_min, min_periods=minp).std().astype(float)

            lower = vwap - args.band_k * sigma
            upper = vwap + args.band_k * sigma

            # per-touch cooldown
            last_touch = -10**9
            for i in range(len(g)):
                if i - last_touch < args.cooldown_min:
                    continue
                vi = float(vwap.iloc[i]) if pd.notna(vwap.iloc[i]) else np.nan
                si = float(sigma.iloc[i]) if pd.notna(sigma.iloc[i]) else np.nan
                li = float(lower.iloc[i]) if pd.notna(lower.iloc[i]) else np.nan
                ui = float(upper.iloc[i]) if pd.notna(upper.iloc[i]) else np.nan
                if not np.isfinite(vi) or not np.isfinite(si) or si <= 1e-9 or not np.isfinite(li) or not np.isfinite(ui):
                    continue

                # dip touch
                if float(g.at[i, "Low"]) <= li:
                    entry = li
                    stop = li - args.stop_sigma_mult * si

                    i1 = min(len(g) - 1, i + args.horizon_min)
                    after = g.iloc[i:i1+1]
                    y = _label_success(after, stop=stop, vwap=vi)

                    lb = max(0, i - args.lookback_min)
                    win = g.iloc[lb:i+1].copy()
                    win_vwap = vwap.iloc[lb:i+1].astype(float).to_numpy()
                    win_close = win["Close"].astype(float).to_numpy()

                    # vwap crosses in lookback
                    above = (win_close > win_vwap).astype(int)
                    crosses = int(np.sum(above[1:] != above[:-1])) if len(above) >= 2 else 0

                    # slope
                    slope = _rolling_slope(win_close)

                    # ATR-ish proxy
                    atr = float(np.nanmean((win["High"].astype(float) - win["Low"].astype(float)).to_numpy()))

                    # relvol
                    v_now = float(g.at[i, "Volume"])
                    relvol5 = v_now / (float(win["Volume"].astype(float).tail(5).mean()) + 1e-12)
                    relvol15 = v_now / (float(win["Volume"].astype(float).tail(15).mean()) + 1e-12)

                    # time of day minutes since open
                    dt_et = g.at[i, "dt"].tz_convert(_ET)
                    tod_min = (dt_et.hour * 60 + dt_et.minute) - (9 * 60 + 30)

                    # daily context (lagged features)
                    key = (sym, d)
                    if key in daily_panel_key.index:
                        ctx = daily_panel_key.loc[key]
                        if isinstance(ctx, pd.DataFrame):
                            ctx = ctx.iloc[-1]
                        trend_20_50 = float(ctx["trend_20_50"]) if pd.notna(ctx["trend_20_50"]) else np.nan
                        vol20 = float(ctx["vol20"]) if pd.notna(ctx["vol20"]) else np.nan
                        avg20_dollar_vol = float(ctx["avg20_dollar_vol"]) if pd.notna(ctx["avg20_dollar_vol"]) else np.nan
                        mom5 = float(ctx["mom5"]) if pd.notna(ctx["mom5"]) else np.nan
                        mom20 = float(ctx["mom20"]) if pd.notna(ctx["mom20"]) else np.nan
                        gap_pct = float(ctx["gap_pct"]) if pd.notna(ctx["gap_pct"]) else np.nan
                        atr14_pct = float(ctx["atr14_pct"]) if pd.notna(ctx["atr14_pct"]) else np.nan
                    else:
                        trend_20_50 = vol20 = avg20_dollar_vol = mom5 = mom20 = gap_pct = atr14_pct = np.nan

                    close_i = float(g.at[i, "Close"])
                    zscore = (close_i - vi) / (si + 1e-12)
                    band_width_pct = (ui - li) / max(1e-12, vi)
                    dist_to_entry_sig = (close_i - entry) / (si + 1e-12)
                    dist_to_stop_sig = (close_i - stop) / (si + 1e-12)

                    rows.append({
                        "symbol": sym,
                        "day": d.isoformat(),
                        "ts_utc": int(g.at[i, "dt"].value),  # ns
                        "label": int(y),

                        # RR intraday features
                        "zscore": float(zscore),
                        "band_width_pct": float(band_width_pct),
                        "vwap_crosses": float(crosses),
                        "slope": float(slope),
                        "atr_proxy": float(atr),
                        "sigma": float(si),
                        "relvol5": float(relvol5),
                        "relvol15": float(relvol15),
                        "tod_min": float(tod_min),
                        "transactions": float(g.at[i, "transactions"]),
                        "volume": float(v_now),

                        # daily context
                        "trend_20_50": trend_20_50,
                        "vol20": vol20,
                        "avg20_dollar_vol": avg20_dollar_vol,
                        "mom5": mom5,
                        "mom20": mom20,
                        "gap_pct": gap_pct,
                        "atr14_pct": atr14_pct,

                        # risk geometry
                        "dist_to_entry_sig": float(dist_to_entry_sig),
                        "dist_to_stop_sig": float(dist_to_stop_sig),
                    })

                    last_touch = i

        if rows:
            df_out = pd.DataFrame(rows)
            day_dir = out_dir / f"day={d.isoformat()}"
            day_dir.mkdir(parents=True, exist_ok=True)
            out_path = day_dir / f"part-{d.isoformat()}.parquet"
            pa, pq = _require_pyarrow()
            table = pa.Table.from_pandas(df_out, preserve_index=False)
            pq.write_table(table, out_path, compression="zstd")
            total_events += len(rows)

        d += timedelta(days=1)

    meta = {
        "out_dir": str(out_dir),
        "total_events": int(total_events),
        "built_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[OK] wrote", meta)


if __name__ == "__main__":
    main()
