from __future__ import annotations
import pandas as pd

def vwap(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("VWAP requires non-empty bars")
    for c in ("High","Low","Close","Volume"):
        if c not in df.columns:
            raise ValueError(f"VWAP requires {c}")
    tp = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3.0
    vol = df["Volume"].astype(float)
    cum_vol = vol.cumsum()
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol


def trend_state_1m(df: pd.DataFrame, vw: pd.Series | None = None, lookback: int = 15) -> dict:
    """Compute a lightweight trend state from 1m bars (tenant-safe: pure math on real bars).

    Returns:
      - state: one of {"up","down","chop","unknown","reclaim_vwap","lost_vwap"}
      - vwap_last: float | None
      - vwap_delta_pct: float | None
      - slope_pct_lookback: float | None   (approx % move per lookback window)
    """
    if df is None or df.empty:
        return {"state": "unknown", "vwap_last": None, "vwap_delta_pct": None, "slope_pct_lookback": None}

    close = None
    if "Close" in df.columns:
        close = df["Close"].astype(float)
    elif "close" in df.columns:
        close = df["close"].astype(float)
    else:
        return {"state": "unknown", "vwap_last": None, "vwap_delta_pct": None, "slope_pct_lookback": None}

    n = int(len(close))
    if n < max(5, lookback + 1):
        # too few bars to classify reliably
        vwap_last = float(vw.iloc[-1]) if (vw is not None and len(vw) > 0) else None
        last = float(close.iloc[-1])
        vwap_delta_pct = ((last - vwap_last) / vwap_last * 100.0) if (vwap_last and vwap_last > 0) else None
        return {"state": "unknown", "vwap_last": vwap_last, "vwap_delta_pct": vwap_delta_pct, "slope_pct_lookback": None}

    # simple slope over last `lookback` bars using endpoints (robust & cheap)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-1 - lookback])
    slope_pct = ((last - prev) / prev * 100.0) if prev > 0 else None

    vwap_last = None
    vwap_delta_pct = None
    above = None
    crossed = None
    if vw is not None and len(vw) == n:
        vwap_last = float(vw.iloc[-1])
        if vwap_last > 0:
            vwap_delta_pct = (last - vwap_last) / vwap_last * 100.0
        above = bool(last > vwap_last)
        # detect recent cross of close vs vwap
        recent = min(10, n-2)
        if recent >= 2:
            past_above = [bool(float(close.iloc[-1-i]) > float(vw.iloc[-1-i])) for i in range(1, recent+1)]
            if above and any(a is False for a in past_above):
                crossed = "reclaim_vwap"
            elif (not above) and any(a is True for a in past_above):
                crossed = "lost_vwap"

    # classify
    state = "chop"
    if slope_pct is None:
        state = "unknown"
    else:
        if above is True and slope_pct > 0.15:
            state = "up"
        elif above is False and slope_pct < -0.15:
            state = "down"
        else:
            state = "chop"

    if crossed is not None:
        state = crossed

    return {"state": state, "vwap_last": vwap_last, "vwap_delta_pct": vwap_delta_pct, "slope_pct_lookback": slope_pct}

def avg_daily_volume(daily: pd.DataFrame, window: int = 20) -> float | None:
    if daily is None or daily.empty or len(daily) < window:
        return None
    if "Volume" not in daily.columns:
        raise ValueError("Volume column missing in daily")
    v = daily["Volume"].astype(float).rolling(window=window, min_periods=window).mean().dropna()
    return float(v.iloc[-1]) if not v.empty else None
