from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from providers.alpaca_provider import AlpacaProvider
from providers.base import BarsRequest
from scanner.indicators import vwap as vwap_series, trend_state_1m, avg_daily_volume

_ET = ZoneInfo("America/New_York")


def _find_col(df: pd.DataFrame, *cands: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _normalize_minute_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Massive minute_aggs_v1 flatfile slice into our OHLCV schema.

    Expected output columns: Open/High/Low/Close/Volume and UTC tz-aware index.
    This function is intentionally defensive: if Massive changes a column name,
    we raise a real exception listing what we saw.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    ts_col = _find_col(df, "timestamp", "time", "t", "window_start", "start", "datetime")
    sym_col = _find_col(df, "symbol", "ticker", "sym", "S")
    o_col = _find_col(df, "open", "o", "Open")
    h_col = _find_col(df, "high", "h", "High")
    l_col = _find_col(df, "low", "l", "Low")
    c_col = _find_col(df, "close", "c", "Close")
    v_col = _find_col(df, "volume", "v", "Volume")

    missing = [
        ("timestamp", ts_col),
        ("symbol", sym_col),
        ("open", o_col),
        ("high", h_col),
        ("low", l_col),
        ("close", c_col),
        ("volume", v_col),
    ]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise RuntimeError(f"flatfile minute schema missing {missing}; columns={list(df.columns)}")

    out = df[[ts_col, o_col, h_col, l_col, c_col, v_col]].copy()
    out.columns = ["ts", "Open", "High", "Low", "Close", "Volume"]

    # timestamp parsing: support ISO strings or integer epoch (s/ms/us/ns)
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
        out.index = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True)
    else:
        out.index = pd.to_datetime(ts, utc=True, errors="raise")

    out = out.drop(columns=["ts"])
    return out
def load_minute_history_from_flatfiles(flat_minute_root: str, *, symbol: str, days_back: int) -> pd.DataFrame:
    """Load 1m history for a symbol from Massive flatfiles.

    Assumes local cache layout mirroring S3:
      <root>/YYYY/MM/YYYY-MM-DD.csv.gz

    Each daily file contains many symbols; we filter to the one requested.
    """
    from pathlib import Path

    root = Path(flat_minute_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"flat_minute_root not found: {root}")

    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    end = datetime.now(tz=_ET).date()
    start = end - timedelta(days=int(days_back))

    frames = []
    for n in range((end - start).days + 1):
        d = start + timedelta(days=n)
        path = root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.isoformat()}.csv.gz"
        if not path.exists():
            continue
        df = pd.read_csv(path, compression="gzip")
        # filter symbol
        sym_col = _find_col(df, "symbol", "ticker", "sym", "S")
        if sym_col is None:
            raise RuntimeError(f"flatfile missing symbol column; columns={list(df.columns)}")
        df = df[df[sym_col].astype(str).str.upper() == symbol]
        if df.empty:
            continue
        frames.append(_normalize_minute_flat(df))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out


def load_daily_history_from_flatfiles(flat_daily_root: str, *, symbol: str) -> pd.DataFrame:
    """Load daily history for a symbol from Massive day_aggs_v1 flatfiles.

    Expected local layout (mirrors S3):
      <root>/YYYY/MM/YYYY-MM-DD.csv.gz
    """
    from pathlib import Path

    root = Path(flat_daily_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"flat_daily_root not found: {root}")
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    frames = []
    for ydir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for mdir in sorted([p for p in ydir.iterdir() if p.is_dir()]):
            for f in sorted(mdir.glob("*.csv.gz")):
                df = pd.read_csv(f, compression="gzip")
                sym_col = _find_col(df, "symbol", "ticker", "sym", "S")
                ts_col = _find_col(df, "date", "day", "timestamp", "t")
                o_col = _find_col(df, "open", "o")
                h_col = _find_col(df, "high", "h")
                l_col = _find_col(df, "low", "l")
                c_col = _find_col(df, "close", "c")
                v_col = _find_col(df, "volume", "v")
                if sym_col is None or ts_col is None or c_col is None or v_col is None:
                    raise RuntimeError(f"flatfile daily schema unexpected; columns={list(df.columns)}")
                df = df[df[sym_col].astype(str).str.upper() == symbol]
                if df.empty:
                    continue
                out = df[[ts_col, o_col, h_col, l_col, c_col, v_col]].copy()
                out.columns = ["ts", "Open", "High", "Low", "Close", "Volume"]
                out.index = pd.to_datetime(out["ts"], utc=True, errors="coerce")
                out = out.drop(columns=["ts"]).dropna(subset=["Close"])
                frames.append(out)

    if not frames:
        return pd.DataFrame()
    daily = pd.concat(frames).sort_index()
    return daily


@dataclass(frozen=True)
class EntryNowParams:
    horizon_min: int = 30              # label horizon
    sample_every_min: int = 3          # stride for sampling minutes
    lookback_trend_min: int = 15       # trend slope window
    swing_lookback_min: int = 15       # swing low/high window
    min_price: float = 1.0
    max_price: float = 30.0
    min_stop_pct: float = 0.25         # reject too-tight stops (percent)
    max_stop_pct: float = 8.0          # reject too-wide stops (percent)
    vwap_buffer_pct: float = 0.10      # buffer around vwap for stop candidate (percent)
    swing_buffer_pct: float = 0.05     # buffer beyond swing for stop candidate (percent)
    include_premarket: bool = False    # RTH-only by default (stable)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if getattr(df.index, "tz", None) is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


def _slice_rth_1m(df_1m: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Slice 1m bars to RTH session (09:30–16:00 ET) for a given ET day."""
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    df = _ensure_utc_index(df_1m)
    df_et = df.tz_convert(_ET)

    d = pd.Timestamp(day)
    if d.tzinfo is None:
        d_et = d.tz_localize(_ET)
    else:
        d_et = d.tz_convert(_ET)

    start = d_et.normalize() + pd.Timedelta(hours=9, minutes=30)
    end = d_et.normalize() + pd.Timedelta(hours=16, minutes=0)
    out = df_et[(df_et.index >= start) & (df_et.index < end)]
    return out.tz_convert("UTC")


def _slice_pm_1m(df_1m: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Slice 1m bars to Premarket session (04:00–09:30 ET) for a given ET day."""
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    df = _ensure_utc_index(df_1m)
    df_et = df.tz_convert(_ET)

    d = pd.Timestamp(day)
    if d.tzinfo is None:
        d_et = d.tz_localize(_ET)
    else:
        d_et = d.tz_convert(_ET)

    start = d_et.normalize() + pd.Timedelta(hours=4, minutes=0)
    end = d_et.normalize() + pd.Timedelta(hours=9, minutes=30)
    out = df_et[(df_et.index >= start) & (df_et.index < end)]
    return out.tz_convert("UTC")



def _opening_range_5m_from_1m(rth_1m: pd.DataFrame) -> Tuple[float, float]:
    """Compute a 5-minute opening range from 1m RTH bars.

    Fail closed unless the slice truly begins at the market open.
    """
    if rth_1m is None or rth_1m.empty or len(rth_1m) < 5:
        raise RuntimeError("Not enough 1m bars for opening range")

    idx = pd.DatetimeIndex(rth_1m.index)
    if idx.tz is None:
        idx = idx.tz_localize(_ET)
    else:
        idx = idx.tz_convert(_ET)

    first_ts = idx[0]
    expected_open = first_ts.normalize() + pd.Timedelta(hours=9, minutes=30)
    # Tolerance: accept any bars that fall within the 09:30–09:35 ET window
    # Strict bar-by-bar timestamp matching fails on sparse data / feed gaps
    window_end = expected_open + pd.Timedelta(minutes=5)
    or_mask = (idx >= expected_open) & (idx < window_end)
    or_bars = rth_1m[or_mask.values if hasattr(or_mask, 'values') else or_mask]
    if len(or_bars) < 3:
        raise RuntimeError("Opening range slice has fewer than 3 bars in 09:30–09:35 ET window")

    first5 = or_bars
    hi = float(first5["High"].astype(float).max())
    lo = float(first5["Low"].astype(float).min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= 0 or lo <= 0:
        raise RuntimeError("Invalid opening range values")
    return hi, lo


def _today_dollar_vol_so_far(rth_1m: pd.DataFrame, end_i: int) -> float:
    w = rth_1m.iloc[: end_i + 1]
    # Use typical price (H+L+C)/3 to match VWAP weighting — more accurate than close×vol
    typ = (w["High"].astype(float) + w["Low"].astype(float) + w["Close"].astype(float)) / 3.0
    vol = w["Volume"].astype(float)
    return float((typ * vol).sum())


def _stop_from_structure(
    *,
    side: str,
    entry: float,
    rth_1m: pd.DataFrame,
    end_i: int,
    vw_last: float | None,
    or_high: float | None,
    or_low: float | None,
    p: EntryNowParams,
) -> float | None:
    """Deterministic stop rule for entry-now.

    Tenant-safe: pure math on real bars up to end_i.
    """
    side_l = str(side).lower()
    is_long = side_l == "long"
    if not np.isfinite(entry) or entry <= 0:
        return None
    look = int(p.swing_lookback_min)
    start_i = max(0, end_i - look)
    window = rth_1m.iloc[start_i : end_i + 1]
    if window.empty:
        return None

    lo = float(window["Low"].astype(float).min())
    hi = float(window["High"].astype(float).max())

    # Candidate stops
    cand = []

    if vw_last is not None and np.isfinite(vw_last) and vw_last > 0:
        buf = (p.vwap_buffer_pct / 100.0) * vw_last
        if is_long:
            cand.append(float(vw_last - buf))
        else:
            cand.append(float(vw_last + buf))

    if is_long:
        buf = (p.swing_buffer_pct / 100.0) * max(1e-9, lo)
        cand.append(float(lo - buf))
        if or_low is not None and np.isfinite(or_low) and or_low > 0:
            cand.append(float(or_low))
        # choose the tightest stop that is still below entry
        cand = [x for x in cand if np.isfinite(x) and x > 0 and x < entry]
        if not cand:
            return None
        stop = max(cand)
    else:
        buf = (p.swing_buffer_pct / 100.0) * max(1e-9, hi)
        cand.append(float(hi + buf))
        if or_high is not None and np.isfinite(or_high) and or_high > 0:
            cand.append(float(or_high))
        cand = [x for x in cand if np.isfinite(x) and x > 0 and x > entry]
        if not cand:
            return None
        stop = min(cand)

    # Reject absurdly tight/wide stops (in pct)
    dist = abs(entry - stop)
    dist_pct = dist / entry * 100.0
    if dist_pct < float(p.min_stop_pct) or dist_pct > float(p.max_stop_pct):
        return None
    return float(stop)


def _hit_order_within_horizon(
    *,
    side: str,
    entry: float,
    stop: float,
    target: float,
    future_1m: pd.DataFrame,
) -> int:
    """Return 1 if target hit before stop within future_1m, else 0.

    Uses bar-level first-touch logic (conservative).
    """
    is_long = str(side).lower() == "long"
    for _, row in future_1m.iterrows():
        hi = float(row["High"])
        lo = float(row["Low"])
        if is_long:
            if lo <= stop:
                return 0
            if hi >= target:
                return 1
        else:
            if hi >= stop:
                return 0
            if lo <= target:
                return 1
    return 0


def build_entry_now_samples_for_symbol(
    *,
    provider: AlpacaProvider,
    symbol: str,
    p: EntryNowParams,
    days_back: int,
    failures: list[dict],
    flat_minute_root: str | None = None,
    flat_daily_root: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X,y) samples for one symbol over recent days.

    Returns X dataframe, y series, and group labels (symbol-day) for splits.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame(), pd.Series(dtype=int), []

    # Daily history for avg20 volume and last close per day (no lookahead)
    daily = None
    if flat_daily_root:
        try:
            daily = load_daily_history_from_flatfiles(flat_daily_root, symbol=symbol)
        except Exception as e:
            failures.append({"symbol": symbol, "stage": "daily_flat", "error": f"{type(e).__name__}: {e}"})
            return pd.DataFrame(), pd.Series(dtype=int), []
    if daily is None:
        try:
            daily = provider.get_daily_history(symbol, period=f"{max(120, days_back + 60)}d").sort_index()
        except Exception as e:
            failures.append({"symbol": symbol, "stage": "daily", "error": f"{type(e).__name__}: {e}"})
            return pd.DataFrame(), pd.Series(dtype=int), []

    if daily is None or daily.empty or len(daily) < 60:
        failures.append({"symbol": symbol, "stage": "daily", "error": "daily_insufficient"})
        return pd.DataFrame(), pd.Series(dtype=int), []

    # Intraday 1m
    intraday = None
    if flat_minute_root:
        try:
            intraday = load_minute_history_from_flatfiles(flat_minute_root, symbol=symbol, days_back=days_back)
        except Exception as e:
            failures.append({"symbol": symbol, "stage": "intraday_flat", "error": f"{type(e).__name__}: {e}"})
            return pd.DataFrame(), pd.Series(dtype=int), []
    if intraday is None:
        try:
            intraday = provider.get_bars(BarsRequest(symbol=symbol, interval="1m", period=f"{days_back}d", include_prepost=p.include_premarket)).sort_index()
        except Exception as e:
            failures.append({"symbol": symbol, "stage": "intraday", "error": f"{type(e).__name__}: {e}"})
            return pd.DataFrame(), pd.Series(dtype=int), []

    if intraday is None or intraday.empty:
        failures.append({"symbol": symbol, "stage": "intraday", "error": "intraday_empty"})
        return pd.DataFrame(), pd.Series(dtype=int), []

    intraday = _ensure_utc_index(intraday)

    # Determine unique ET session dates present in intraday
    intraday_et = intraday.tz_convert(_ET)
    days = sorted({ts.date() for ts in intraday_et.index})
    if not days:
        return pd.DataFrame(), pd.Series(dtype=int), []

    # Focus on the last N calendar days
    days = days[-days_back:]

    rows = []
    ys = []
    groups = []

    for d in days:
        day_ts = pd.Timestamp(datetime(d.year, d.month, d.day), tz=_ET)
        rth = _slice_rth_1m(intraday, day_ts)
        if rth is None or rth.empty or len(rth) < 60:
            continue

        # avoid lookahead: daily up to previous day
        day_utc = rth.index[0].normalize()
        daily_prior = daily[daily.index.normalize() < day_utc]
        if len(daily_prior) < 60:
            continue

        # price filter uses entry-at-t, so we cannot prefilter symbol here reliably
        # compute avg20 dollar vol on prior close
        try:
            avg20_vol = avg_daily_volume(daily_prior, window=20)
            prev_close = float(daily_prior["Close"].astype(float).iloc[-1])
            avg20_dollar_vol = float(avg20_vol * prev_close) if (avg20_vol and prev_close > 0) else None
        except Exception:
            avg20_dollar_vol = None

        # Precompute VWAP series for the day
        try:
            vw = vwap_series(rth)
        except Exception:
            vw = None

        # OR from first 5m of 1m bars
        try:
            or_high, or_low = _opening_range_5m_from_1m(rth)
        except Exception:
            or_high = or_low = None

        stride = max(1, int(p.sample_every_min))
        horizon_bars = max(1, int(p.horizon_min))
        # sample from 09:35 to 11:30 ET by index positions (skip first 5m)
        start_i = 5
        end_i_max = min(len(rth) - horizon_bars - 1, 120)  # ~ first 2 hours
        for i in range(start_i, end_i_max, stride):
            entry = float(rth["Close"].astype(float).iloc[i])
            if not np.isfinite(entry) or entry <= 0:
                continue
            if entry < float(p.min_price) or entry > float(p.max_price):
                continue

            # trend state using bars up to i
            try:
                sub = rth.iloc[: i + 1]
                sub_vw = vw.iloc[: i + 1] if (vw is not None) else None
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

            # simple volume features up to i
            try:
                dv_so_far = _today_dollar_vol_so_far(rth, i)
            except Exception:
                dv_so_far = None

            # RVOL approx: today vol so far / avg20 daily vol (scaled by fraction of day elapsed)
            try:
                today_vol_so_far = float(rth["Volume"].astype(float).iloc[: i + 1].sum())
                avg20_vol = float(avg_daily_volume(daily_prior, window=20) or 0.0)
                frac = max(1e-6, (i + 1) / float(len(rth)))
                rvol_now = (today_vol_so_far / max(1e-9, avg20_vol * frac)) if avg20_vol > 0 else None
            except Exception:
                rvol_now = None

            # distances
            dist_orh_pct = None
            dist_orl_pct = None
            or_range_pct = None
            if or_high and or_low:
                mid = (or_high + or_low) / 2.0
                or_range_pct = ((or_high - or_low) / max(1e-9, mid)) * 100.0
                dist_orh_pct = ((entry - or_high) / max(1e-9, or_high)) * 100.0
                dist_orl_pct = ((entry - or_low) / max(1e-9, or_low)) * 100.0

            # build two directional samples (long + short) when valid stop exists
            for side in ("long", "short"):
                stop = _stop_from_structure(
                    side=side,
                    entry=entry,
                    rth_1m=rth,
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

                # label
                future = rth.iloc[i + 1 : i + 1 + horizon_bars]
                y = _hit_order_within_horizon(side=side, entry=entry, stop=float(stop), target=float(target), future_1m=future)

                rows.append({
                    "side_long": 1.0 if side == "long" else 0.0,
                    "entry": entry,
                    "stop_dist_pct": (R / entry) * 100.0,
                    "vwap_delta_pct": float(vwap_delta_pct) if vwap_delta_pct is not None else np.nan,
                    "trend_slope_pct": float(slope) if slope is not None else np.nan,
                    "trend_state_up": 1.0 if tstate in ("up","reclaim_vwap") else 0.0,
                    "trend_state_down": 1.0 if tstate in ("down","lost_vwap") else 0.0,
                    "trend_state_chop": 1.0 if tstate == "chop" else 0.0,
                    "or_range_pct": float(or_range_pct) if or_range_pct is not None else np.nan,
                    "dist_orh_pct": float(dist_orh_pct) if dist_orh_pct is not None else np.nan,
                    "dist_orl_pct": float(dist_orl_pct) if dist_orl_pct is not None else np.nan,
                    "today_dollar_vol_so_far": float(dv_so_far) if dv_so_far is not None else np.nan,
                    "avg20_dollar_vol": float(avg20_dollar_vol) if avg20_dollar_vol is not None else np.nan,
                    "rvol_now": float(rvol_now) if rvol_now is not None else np.nan,
                    "minutes_since_open": float(i),
                })
                ys.append(int(y))
                groups.append(f"{symbol}:{d.isoformat()}")
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), []
    X = pd.DataFrame(rows)
    y = pd.Series(ys, dtype=int)
    return X, y, groups
