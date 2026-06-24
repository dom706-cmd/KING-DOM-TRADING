from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _normalize_symbol(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol_required")
    return sym


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _payload_age_seconds(ts_value: Any) -> float | None:
    dt = _parse_timestamp(ts_value)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def latest_trade_payload(stream_cache: Any, symbol: str, *, max_age_sec: float | None = None) -> dict[str, Any]:
    if stream_cache is None or not hasattr(stream_cache, "latest_trade"):
        raise RuntimeError("stream_cache_missing_trade_reader")
    sym = _normalize_symbol(symbol)
    try:
        payload = stream_cache.latest_trade(sym) or {}
    except Exception as e:
        raise RuntimeError(f"stream_trade_read_failed:{type(e).__name__}:{e}") from e
    if not payload:
        raise RuntimeError(f"stream_trade_missing:{sym}")
    price = payload.get("price")
    if price is None:
        raise RuntimeError(f"stream_trade_missing_price:{sym}")
    age = _payload_age_seconds(payload.get("timestamp"))
    if max_age_sec is not None:
        if age is None:
            raise RuntimeError(f"stream_trade_missing_timestamp:{sym}")
        if age > float(max_age_sec):
            raise RuntimeError(f"stream_trade_stale:{sym}:{age:.3f}s")
    out = dict(payload)
    out["price"] = float(price)
    out["age_sec"] = age
    return out


def latest_quote_payload(stream_cache: Any, symbol: str, *, max_age_sec: float | None = None) -> dict[str, Any]:
    if stream_cache is None or not hasattr(stream_cache, "latest_quote"):
        raise RuntimeError("stream_cache_missing_quote_reader")
    sym = _normalize_symbol(symbol)
    try:
        payload = stream_cache.latest_quote(sym) or {}
    except Exception as e:
        raise RuntimeError(f"stream_quote_read_failed:{type(e).__name__}:{e}") from e
    if not payload:
        raise RuntimeError(f"stream_quote_missing:{sym}")
    bid = payload.get("bid_price", payload.get("bid"))
    ask = payload.get("ask_price", payload.get("ask"))
    if bid is None or ask is None:
        raise RuntimeError(f"stream_quote_missing_bid_ask:{sym}")
    age = _payload_age_seconds(payload.get("timestamp") or payload.get("quote_ts"))
    if max_age_sec is not None:
        if age is None:
            raise RuntimeError(f"stream_quote_missing_timestamp:{sym}")
        if age > float(max_age_sec):
            raise RuntimeError(f"stream_quote_stale:{sym}:{age:.3f}s")
    out = dict(payload)
    out["bid"] = float(bid)
    out["ask"] = float(ask)
    out["age_sec"] = age
    return out


def recent_bars_df(
    stream_cache: Any,
    symbol: str,
    *,
    session_date: date | None = None,
    regular_hours_only: bool = True,
    min_bars: int = 1,
) -> pd.DataFrame:
    if stream_cache is None or not hasattr(stream_cache, "recent_1m_bars"):
        raise RuntimeError("stream_cache_missing_bar_reader")
    sym = _normalize_symbol(symbol)
    try:
        rows = list(stream_cache.recent_1m_bars(sym) or [])
    except Exception as e:
        raise RuntimeError(f"stream_bars_read_failed:{type(e).__name__}:{e}") from e
    if not rows:
        raise RuntimeError(f"stream_bars_missing:{sym}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"stream_bars_empty:{sym}")

    col_map = {"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "n": "Trades", "vw": "VWAP"}
    missing = [k for k in ("t", "o", "h", "l", "c", "v") if k not in df.columns]
    if missing:
        raise RuntimeError(f"stream_bars_missing_columns:{sym}:{','.join(missing)}")

    ts = pd.to_datetime(df["t"], utc=True, errors="coerce")
    if ts.isna().all():
        raise RuntimeError(f"stream_bars_invalid_timestamp:{sym}")

    out = pd.DataFrame(index=ts)
    for src, dst in col_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy()
    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if out.empty:
        raise RuntimeError(f"stream_bars_no_numeric_rows:{sym}")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.tz_convert(ET)

    target_day = session_date or datetime.now(timezone.utc).astimezone(ET).date()
    out = out.loc[out.index.date == target_day]
    if regular_hours_only:
        out = out.between_time("09:30", "16:00", inclusive="both")
    if out.empty:
        raise RuntimeError(f"stream_bars_session_empty:{sym}:{target_day.isoformat()}")
    if len(out) < int(min_bars):
        raise RuntimeError(f"stream_bars_not_enough:{sym}:{len(out)}<{int(min_bars)}")
    # Real-time guard (flag, not reject): expose the age of the most recent bar so
    # callers can mark a candidate stale if the stream has stalled. 1m bars print
    # ~once/minute, so ~90s during RTH indicates a stall rather than normal cadence.
    try:
        _last_ts = out.index[-1].to_pydatetime().astimezone(timezone.utc)
        _age = max(0.0, (datetime.now(timezone.utc) - _last_ts).total_seconds())
        _now_et = datetime.now(timezone.utc).astimezone(ET)
        _is_rth = _now_et.weekday() < 5 and (9 * 60 + 30) <= (_now_et.hour * 60 + _now_et.minute) < (16 * 60)
        out.attrs["last_bar_age_sec"] = round(_age, 1)
        out.attrs["bars_stale"] = bool(_is_rth and _age > 90.0)
    except Exception:
        out.attrs["last_bar_age_sec"] = None
        out.attrs["bars_stale"] = False
    return out
