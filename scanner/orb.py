from __future__ import annotations

from zoneinfo import ZoneInfo
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, time as dtime, date, timedelta, timezone

import math
import pandas as pd

from sentiment.multi import SentimentService
from sentiment.catalyst import CatalystService
from ml.orb_model_service import score_orb_candidates
from core.errors import PlanBuildFailure, failure_dict
from ml.range_reversion_gold import RangeReversionGoldScorer
from providers.base import BarsRequest
from providers.alpaca_provider import AlpacaProvider
from providers.symbols import to_provider_symbol
from .indicators import vwap, avg_daily_volume, trend_state_1m
from runtime.stream_market_data import recent_bars_df, latest_trade_payload, latest_quote_payload

# Resolve project root (…/orb_10.55.1) regardless of where the process is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

ET = ZoneInfo("America/New_York")
_ET = ET

@dataclass(frozen=True)
class ORBConfig:
    min_price: float = 1.0
    max_price: float = 20.0

    # Liquidity / activity
    min_today_dollar_vol: float = 2_000_000.0
    min_avg20_dollar_vol: float = 1_000_000.0
    min_rvol: float = 1.5

    # Prefilter history window (reduce API load; ORB uses intraday for final checks)
    prefilter_period: str = "3mo"

    # OR quality bounds (%)
    min_or_range_pct: float = 0.6
    max_or_range_pct: float = 6.0

    # Risk sizing
    risk_dollars: float = 50.0
    min_risk_per_share: float = 0.06
    max_risk_per_share: float = 0.35
    min_shares: int = 1
    max_notional: float = 100000.0

    # buffer rule
    buffer_min_dollars: float = 0.01
    buffer_pct: float = 0.001

@dataclass
class Candidate:
    symbol: str
    data_date: str
    last_price: float
    pct_change: float | None
    rvol: float | None
    today_dollar_vol: float | None
    avg20_dollar_vol: float | None
    or_high: float
    or_low: float
    or_range_pct: float
    above_vwap: bool | None
    vwap_last: float | None
    vwap_delta_pct: float | None
    trend_state: str | None
    trend_slope_pct: float | None

    # Best-side plan (kept for backward compatibility)
    best_side: str  # "long" or "short"
    entry: float
    stop: float
    target_2r: float
    target_3r: float
    risk_per_share: float
    shares: int
    notional: float

    # Dual-side plans for UI (may be None if invalid)
    long_entry: float | None
    long_stop: float | None
    long_2r: float | None
    long_3r: float | None
    long_risk_per_share: float | None
    long_shares: int | None
    long_notional: float | None

    short_entry: float | None
    short_stop: float | None
    short_2r: float | None
    short_3r: float | None
    short_risk_per_share: float | None
    short_shares: int | None
    short_notional: float | None

    # Trigger context (computed from last_price vs trigger)
    long_triggered: bool | None = None
    short_triggered: bool | None = None
    long_trigger_delta: float | None = None   # trigger - last (>$0 means needs to rise)
    short_trigger_delta: float | None = None  # last - trigger (>$0 means above trigger)

    # Free-form notes (default required because fields above have defaults)
    notes: str = ""
    # Optional enrichment
    ml_score: float | None = None
    model_bucket: str | None = None
    sentiment_score: float | None = None
    catalyst_score: float | None = None
    catalyst_confidence: float | None = None
    catalyst_strength: float | None = None
    catalyst_article_count: int | None = None
    catalyst_freshness_hours: float | None = None
    catalyst_tags: list[str] | None = None
    confidence_score: float | None = None
    confidence_grade: str | None = None
    regime_profile: str | None = None
    regime_adjustment: float | None = None
    combined_score: float | None = None
    # Proof / UI convenience
    scan_date: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy: str | None = None
    exec_style: str | None = None
    gate_passes: bool | None = None
    gate_fail_reasons: list[str] | None = None
    discovery_passes: bool | None = None
    discovery_fail_reasons: list[str] | None = None
    trade_ready_passes: bool | None = None
    trade_ready_fail_reasons: list[str] | None = None
    score_breakdown: dict[str, float | str | bool | None] | None = None
    rr_touch_age_min: float | None = None
    rr_chase_r: float | None = None
    rr_actionable_now: bool | None = None
    scan_ts: str | None = None
    touch_ts: str | None = None
    live_price: float | None = None
    live_price_ts: str | None = None
    live_bid: float | None = None
    live_ask: float | None = None
    live_mid: float | None = None
    live_feed: str | None = None
    stale_reason: str | None = None
    prior_session_touch: bool | None = None
    live_vwap_delta_pct: float | None = None
    live_chase_r: float | None = None
    orb_retest_ready: bool | None = None
    orb_retest_distance_r: float | None = None
    orb_retest_touch_ts: str | None = None
    monitor_seed: bool | None = None
    monitor_seed_reasons: list[str] | None = None
    tradable_now: bool | None = None
    prev_close: float | None = None
    prev_day_high: float | None = None
    recent_20d_high: float | None = None
    dist_from_prev_close_pct: float | None = None
    prev_day_change_pct: float | None = None
    dist_above_prev_day_high_pct: float | None = None
    dist_above_20d_high_pct: float | None = None



def _stream_confidence_grade(score_0_100: float | None) -> str | None:
    if score_0_100 is None:
        return None
    s = float(score_0_100)
    if s >= 85.0:
        return "A"
    if s >= 75.0:
        return "B"
    if s >= 65.0:
        return "C"
    if s >= 55.0:
        return "D"
    return "F"


def _stream_orb_candidate(stream_cache: Any, symbol: str, cfg: ORBConfig, *, session_date: date, exec_style: str = "retest") -> Candidate:
    bars = recent_bars_df(stream_cache, symbol, session_date=session_date, regular_hours_only=True, min_bars=5)
    trade = latest_trade_payload(stream_cache, symbol, max_age_sec=45.0)
    quote = latest_quote_payload(stream_cache, symbol, max_age_sec=45.0)

    last_price = float(trade["price"])
    if not (cfg.min_price <= last_price <= cfg.max_price):
        raise ValueError("filtered_price")

    first_close = float(bars["Close"].astype(float).iloc[0])
    pct_change = ((last_price - first_close) / first_close * 100.0) if first_close > 0 else None
    today_vol = float(bars["Volume"].astype(float).sum())
    today_dollar_vol = float(today_vol * last_price)

    first5 = bars.iloc[:5]
    or_high = float(first5["High"].astype(float).max())
    or_low = float(first5["Low"].astype(float).min())
    or_mid = (or_high + or_low) / 2.0
    if not (or_high > 0 and or_low > 0 and or_high > or_low):
        raise RuntimeError("intraday_invalid_or_values")
    or_range_pct = ((or_high - or_low) / or_mid * 100.0) if or_mid > 0 else 999.0
    if not (cfg.min_or_range_pct <= or_range_pct <= cfg.max_or_range_pct):
        raise ValueError("filtered_or_range")

    vw = vwap(bars)
    vwap_last = float(vw.iloc[-1])
    above_vwap = bool(last_price >= vwap_last)
    tr = trend_state_1m(bars, vw=vw, lookback=15)
    trend_state = str(tr.get("state")) if tr else None
    trend_slope_pct = float(tr.get("slope_pct_lookback")) if tr and tr.get("slope_pct_lookback") is not None else None
    vwap_delta_pct = ((last_price - vwap_last) / vwap_last * 100.0) if vwap_last > 0 else None

    buf_h = _buffer(or_high, cfg)
    buf_l = _buffer(or_low, cfg)

    long_entry = or_high + buf_h
    long_stop = or_low - buf_l
    long_risk = long_entry - long_stop
    short_entry = or_low - buf_l
    short_stop = or_high + buf_h
    short_risk = short_stop - short_entry

    long_valid = long_risk > 0 and (cfg.min_risk_per_share <= long_risk <= cfg.max_risk_per_share)
    short_valid = short_risk > 0 and (cfg.min_risk_per_share <= short_risk <= cfg.max_risk_per_share)

    long_shares = int(math.floor(cfg.risk_dollars / max(long_risk, 1e-9))) if long_valid else 0
    short_shares = int(math.floor(cfg.risk_dollars / max(short_risk, 1e-9))) if short_valid else 0
    long_notional = float(long_shares * long_entry) if long_valid else 0.0
    short_notional = float(short_shares * short_entry) if short_valid else 0.0

    if long_valid and (long_shares < cfg.min_shares or long_notional > cfg.max_notional):
        long_valid = False
    if short_valid and (short_shares < cfg.min_shares or short_notional > cfg.max_notional):
        short_valid = False
    if not long_valid and not short_valid:
        raise ValueError("filtered_no_valid_plan")

    if long_valid and not short_valid:
        best_side = "long"
    elif short_valid and not long_valid:
        best_side = "short"
    else:
        best_side = "long" if above_vwap else "short"

    entry = float(long_entry if best_side == "long" else short_entry)
    stop = float(long_stop if best_side == "long" else short_stop)
    risk = float(long_risk if best_side == "long" else short_risk)
    shares = int(long_shares if best_side == "long" else short_shares)
    notional = float(long_notional if best_side == "long" else short_notional)
    target_2r = float(entry + 2.0 * risk) if best_side == "long" else float(entry - 2.0 * risk)
    target_3r = float(entry + 3.0 * risk) if best_side == "long" else float(entry - 3.0 * risk)
    long_2r = float(long_entry + 2.0 * long_risk) if long_valid else None
    long_3r = float(long_entry + 3.0 * long_risk) if long_valid else None
    short_2r = float(short_entry - 2.0 * short_risk) if short_valid else None
    short_3r = float(short_entry - 3.0 * short_risk) if short_valid else None

    c = Candidate(
        symbol=symbol,
        data_date=session_date.isoformat(),
        last_price=last_price,
        pct_change=pct_change,
        rvol=None,
        today_dollar_vol=today_dollar_vol,
        avg20_dollar_vol=None,
        or_high=or_high,
        or_low=or_low,
        or_range_pct=or_range_pct,
        above_vwap=above_vwap,
        vwap_last=vwap_last,
        vwap_delta_pct=vwap_delta_pct,
        trend_state=trend_state,
        trend_slope_pct=trend_slope_pct,
        best_side=best_side,
        entry=entry,
        stop=stop,
        target_2r=target_2r,
        target_3r=target_3r,
        risk_per_share=risk,
        shares=shares,
        notional=notional,
        long_entry=float(long_entry) if long_valid else None,
        long_stop=float(long_stop) if long_valid else None,
        long_2r=long_2r,
        long_3r=long_3r,
        long_risk_per_share=float(long_risk) if long_valid else None,
        long_shares=int(long_shares) if long_valid else None,
        long_notional=float(long_notional) if long_valid else None,
        short_entry=float(short_entry) if short_valid else None,
        short_stop=float(short_stop) if short_valid else None,
        short_2r=short_2r,
        short_3r=short_3r,
        short_risk_per_share=float(short_risk) if short_valid else None,
        short_shares=int(short_shares) if short_valid else None,
        short_notional=float(short_notional) if short_valid else None,
        long_triggered=bool(last_price >= long_entry) if long_valid else None,
        short_triggered=bool(last_price <= short_entry) if short_valid else None,
        long_trigger_delta=(float(long_entry - last_price) if long_valid else None),
        short_trigger_delta=(float(last_price - short_entry) if short_valid else None),
        notes="streaming_only_runtime_scan",
        strategy="orb",
        exec_style=exec_style,
        scan_date=session_date.isoformat(),
        stop_loss=float(stop),
        take_profit=float(target_2r),
        scan_ts=datetime.now(timezone.utc).isoformat(),
        live_price=float(last_price),
        live_price_ts=trade.get("timestamp"),
        live_bid=float(quote["bid"]),
        live_ask=float(quote["ask"]),
        live_mid=((float(quote["bid"]) + float(quote["ask"])) / 2.0),
        live_feed="stream",
        stale_reason=None,
        orb_retest_ready=bool(abs((last_price - entry) / max(risk, 1e-9)) <= 0.20),
        orb_retest_distance_r=(float((entry - last_price) / max(risk, 1e-9)) if best_side == "long" else float((last_price - entry) / max(risk, 1e-9))),
        monitor_seed=True,
        tradable_now=((best_side == "long" and last_price >= entry) or (best_side == "short" and last_price <= entry)),
    )
    base_rule = _candidate_setup_rule_score(c)
    c.combined_score = float(base_rule)
    c.confidence_score = float(round(base_rule * 100.0, 2))
    c.confidence_grade = _stream_confidence_grade(c.confidence_score)
    c.monitor_seed_reasons = ["stream_live"]
    c.trade_ready_passes = bool(c.tradable_now)
    c.discovery_passes = True
    c.gate_passes = True
    return c


def _stream_rr_candidate(
    stream_cache: Any,
    symbol: str,
    cfg: ORBConfig,
    *,
    session_date: date,
    range_window_min: int = 60,
    band_k: float = 2.0,
    stop_sigma_mult: float = 0.75,
    touch_lookback_min: int = 15,
) -> Candidate:
    df = recent_bars_df(stream_cache, symbol, session_date=session_date, regular_hours_only=True, min_bars=max(45, int(range_window_min)))
    trade = latest_trade_payload(stream_cache, symbol, max_age_sec=45.0)
    quote = latest_quote_payload(stream_cache, symbol, max_age_sec=45.0)

    last_price = float(trade["price"])
    session_open = float(df["Open"].astype(float).iloc[0])

    tp = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3.0
    vol = df["Volume"].astype(float)
    minp = max(10, int(range_window_min // 3))
    vwap_roll = (tp * vol).rolling(range_window_min, min_periods=minp).sum() / (vol.rolling(range_window_min, min_periods=minp).sum() + 1e-12)
    sigma_roll = df["Close"].astype(float).rolling(range_window_min, min_periods=minp).std()
    lower = vwap_roll - float(band_k) * sigma_roll
    upper = vwap_roll + float(band_k) * sigma_roll
    if vwap_roll.dropna().empty or sigma_roll.dropna().empty:
        raise ValueError("rr_not_enough_rth_bars")

    now_ts = df.index[-1]
    cutoff = now_ts - timedelta(minutes=max(int(touch_lookback_min), min(int(range_window_min), 45)))
    low_s = df["Low"].astype(float)
    close_s = df["Close"].astype(float)
    lower_s = lower.astype(float)
    strict_touch_idx = df.index[(low_s <= lower_s) & (df.index >= cutoff)]
    if len(strict_touch_idx) == 0:
        raise ValueError("no_recent_touch")
    t_touch = strict_touch_idx[-1]
    i = int(df.index.get_loc(t_touch))

    vi = float(vwap_roll.iloc[i])
    si = float(sigma_roll.iloc[i])
    ui = float(upper.iloc[i])
    li = float(lower.iloc[i])

    entry = float(close_s.iloc[i])
    stop = float(entry - max(si * float(stop_sigma_mult), 0.01))
    risk = float(entry - stop)
    if not (cfg.min_risk_per_share <= risk <= max(cfg.max_risk_per_share, risk)):
        raise ValueError("risk_out_of_bounds")

    shares = int(math.floor(cfg.risk_dollars / max(risk, 1e-9)))
    notional = float(shares * entry)
    if shares < cfg.min_shares or notional > cfg.max_notional:
        raise ValueError("position_sizing_invalid")

    vw = vwap(df)
    tstate = trend_state_1m(df, vw=vw, lookback=15)
    band_width_pct = float((ui - li) / max(1e-12, vi) * 100.0)

    c = Candidate(
        symbol=symbol,
        data_date=session_date.isoformat(),
        last_price=last_price,
        pct_change=((last_price - session_open) / session_open * 100.0) if session_open > 0 else None,
        rvol=None,
        today_dollar_vol=float(df["Volume"].astype(float).sum() * last_price),
        avg20_dollar_vol=None,
        or_high=ui,
        or_low=li,
        or_range_pct=band_width_pct,
        above_vwap=bool(last_price >= float(vw.iloc[-1])),
        vwap_last=float(vw.iloc[-1]),
        vwap_delta_pct=((last_price - float(vw.iloc[-1])) / float(vw.iloc[-1]) * 100.0) if float(vw.iloc[-1]) > 0 else None,
        trend_state=str(tstate.get("state")) if tstate else None,
        trend_slope_pct=float(tstate.get("slope_pct_lookback")) if tstate and tstate.get("slope_pct_lookback") is not None else None,
        best_side="long",
        entry=entry,
        stop=stop,
        target_2r=float(entry + 2.0 * risk),
        target_3r=float(ui),
        risk_per_share=risk,
        shares=shares,
        notional=notional,
        long_entry=entry,
        long_stop=stop,
        long_2r=float(entry + 2.0 * risk),
        long_3r=float(ui),
        long_risk_per_share=risk,
        long_shares=shares,
        long_notional=notional,
        short_entry=None,
        short_stop=None,
        short_2r=None,
        short_3r=None,
        short_risk_per_share=None,
        short_shares=None,
        short_notional=None,
        notes="streaming_only_runtime_scan",
        strategy="range_reversion",
        exec_style="reversion",
        scan_date=session_date.isoformat(),
        stop_loss=float(stop),
        take_profit=float(entry + 2.0 * risk),
        rr_touch_age_min=float((now_ts - t_touch).total_seconds() / 60.0),
        rr_chase_r=float((last_price - entry) / max(risk, 1e-9)),
        rr_actionable_now=bool(last_price >= entry and last_price <= float(entry + 0.35 * risk)),
        scan_ts=datetime.now(timezone.utc).isoformat(),
        touch_ts=t_touch.tz_convert(timezone.utc).isoformat() if hasattr(t_touch, "tz_convert") else str(t_touch),
        live_price=float(last_price),
        live_price_ts=trade.get("timestamp"),
        live_bid=float(quote["bid"]),
        live_ask=float(quote["ask"]),
        live_mid=((float(quote["bid"]) + float(quote["ask"])) / 2.0),
        live_feed="stream",
        stale_reason=None,
        monitor_seed=True,
        tradable_now=bool(last_price >= entry),
    )
    base_rule = _candidate_setup_rule_score(c)
    c.combined_score = float(base_rule)
    c.confidence_score = float(round(base_rule * 100.0, 2))
    c.confidence_grade = _stream_confidence_grade(c.confidence_score)
    c.monitor_seed_reasons = ["stream_live", "recent_touch"]
    c.trade_ready_passes = bool(c.tradable_now)
    c.discovery_passes = True
    c.gate_passes = True
    return c


def _scan_symbols_streaming_only(
    symbols: List[str],
    cfg: ORBConfig,
    *,
    limit: int,
    stream_cache: Any,
    strategy: str,
    exec_style: str,
    use_ml: bool,
    range_window_min: int = 60,
    band_k: float = 2.0,
    stop_sigma_mult: float = 0.75,
    touch_lookback_min: int = 15,
) -> Dict[str, Any]:
    if stream_cache is None:
        raise RuntimeError("stream_cache_required_for_runtime_scan")
    session_date = datetime.now(timezone.utc).astimezone(ET).date()
    try:
        stream_cache.start()
    except Exception:
        pass
    try:
        stream_cache.ensure_symbols(symbols)
    except Exception as e:
        raise RuntimeError(f"stream_subscription_failed:{type(e).__name__}:{e}") from e

    candidates: List[Candidate] = []
    data_failures: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    prefilter_samples: List[Dict[str, Any]] = []

    builder = _stream_rr_candidate if strategy in {"range_reversion", "rr", "range"} else _stream_orb_candidate

    for sym in [str(s).strip().upper() for s in symbols if str(s).strip()]:
        try:
            if builder is _stream_rr_candidate:
                c = builder(
                    stream_cache,
                    sym,
                    cfg,
                    session_date=session_date,
                    range_window_min=int(range_window_min),
                    band_k=float(band_k),
                    stop_sigma_mult=float(stop_sigma_mult),
                    touch_lookback_min=int(touch_lookback_min),
                )
            else:
                c = builder(stream_cache, sym, cfg, session_date=session_date, exec_style=exec_style)
            candidates.append(c)
            prefilter_samples.append({"symbol": sym, "passed": True, "reason": "stream_live"})
        except Exception as e:
            code, detail = _classify_intraday_exception(e)
            reject_counts[code] = reject_counts.get(code, 0) + 1
            item = {
                "symbol": sym,
                "stage": "stream_scan",
                "error": detail,
                "code": code,
                "session_date": session_date.isoformat(),
                "strategy": strategy,
                "exec_style": exec_style,
            }
            data_failures.append(item)
            if len(prefilter_samples) < 10:
                prefilter_samples.append({"symbol": sym, "passed": False, "reason": code})

    if use_ml:
        data_failures.append({
            "symbol": "__all__",
            "stage": "ml",
            "error": "stream_only_runtime_scan_has_no_historical_feature_fetch",
            "code": "ml_unavailable_stream_only",
        })

    candidates.sort(key=lambda c: (float(c.tradable_now or 0.0), float(c.combined_score or 0.0), float(c.today_dollar_vol or 0.0)), reverse=True)
    top = candidates[: int(limit)]
    failure_samples_by_code = _build_failure_samples_by_code(data_failures, limit_per_code=5)
    pre_counts = {
        "stream_requested": len(symbols),
        "stream_candidates": len(candidates),
        "stream_failures": len(data_failures),
    }
    return {
        "provider": "alpaca_stream",
        "scan_date": session_date.isoformat(),
        "regime": {"selected": "stream_live"},
        "debug": {
            "session_date_used": session_date.isoformat(),
            "failure_samples": data_failures[:10],
            "failure_samples_by_code": failure_samples_by_code,
            "streaming_only": True,
        },
        "count": len(top),
        "candidates_total": len(candidates),
        "discovery_total": len(candidates),
        "seed_candidates_total": len(candidates),
        "tradable_now_total": sum(1 for c in candidates if c.tradable_now),
        "trade_ready_total": sum(1 for c in candidates if c.trade_ready_passes),
        "rejected_total": len(data_failures),
        "candidates": [({**asdict(c), "price": float(c.last_price or 0.0), "scan_date": session_date.isoformat(), "scan_ts": (c.scan_ts or datetime.now(timezone.utc).isoformat())}) for c in top],
        "rejected_candidates": [],
        "prefilter_counts": pre_counts,
        "prefilter_samples": prefilter_samples[:10],
        "thresholds_used": {"streaming_only": True, "monitor_first_mode": True},
        "reject_counts": reject_counts,
        "data_failures": data_failures,
        "shortlisted": len(symbols),
    }


def _buffer(price: float, cfg: ORBConfig) -> float:
    return max(cfg.buffer_min_dollars, cfg.buffer_pct * price)


def _get_opening_range_5m(intraday_1m: pd.DataFrame) -> Tuple[float, float, str]:
    """
    Derive a truthful opening range from the first available real 1-minute RTH bars.

    Monitor-first rules:
    - never invent bars
    - prefer a true 09:30 ET print
    - if 09:30 is missing, allow the earliest observed RTH print through 09:45
    - tolerate modest real provider gaps
    - if only 3-4 valid opening bars exist, use them as a provisional OR seed
    """
    df = intraday_1m.sort_index()

    if getattr(df.index, "tz", None) is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        df = df.tz_convert(ET)

    missing_cols = _looks_like_missing_columns(df, ("Open", "High", "Low", "Close", "Volume"))
    if missing_cols:
        raise RuntimeError(f"intraday_missing_columns:{missing_cols}")

    day = df.index[-1].date()
    data_date = day.isoformat()

    session = df.loc[df.index.date == day]
    session = session.between_time("09:30", "16:00", inclusive="both")
    if session.empty:
        raise RuntimeError("intraday_or_window")

    session = session[~session.index.duplicated(keep="first")].sort_index()

    nominal_open = datetime.combine(day, dtime(9, 30), tzinfo=ET)
    opening_band = session.loc[
        (session.index >= nominal_open) &
        (session.index <= nominal_open + pd.Timedelta(minutes=15))
    ].copy()

    if opening_band.empty:
        first_ts = session.index[0].isoformat() if len(session.index) else "none"
        raise RuntimeError(f"intraday_or_window_missing_open:first_rth_bar={first_ts}")

    anchor_ts = opening_band.index[0]
    anchor_delay_min = int((anchor_ts - nominal_open).total_seconds() // 60)
    if anchor_delay_min > 15:
        raise RuntimeError(f"intraday_or_window_missing_open:anchor_delay_min={anchor_delay_min}")

    candidate = session.loc[
        (session.index >= anchor_ts) &
        (session.index <= anchor_ts + pd.Timedelta(minutes=35))
    ].copy()

    if len(candidate.index) < 3:
        raise RuntimeError(f"intraday_or_window_incomplete:candidate_bars={len(candidate.index)}")

    window = candidate.iloc[: min(5, len(candidate.index))].copy()
    if len(window.index) < 3:
        raise RuntimeError(f"intraday_or_window_incomplete:window_bars={len(window.index)}")

    anchor_offsets = [int((ts - anchor_ts).total_seconds() // 60) for ts in window.index]
    gaps = [b - a for a, b in zip(anchor_offsets, anchor_offsets[1:])]

    gap_gt6 = sum(1 for g in gaps if g > 6)
    if any(g <= 0 or g > 10 for g in gaps) or gap_gt6 > 2:
        raise RuntimeError(f"intraday_or_window_incomplete:gaps={gaps}")

    if anchor_offsets[-1] > 25:
        raise RuntimeError(f"intraday_or_window_incomplete:last_offset={anchor_offsets[-1]}")

    or_high = float(window["High"].astype(float).max())
    or_low = float(window["Low"].astype(float).min())
    if not math.isfinite(or_high) or not math.isfinite(or_low) or or_low >= or_high:
        raise RuntimeError(f"intraday_invalid_or_values:or_high={or_high}:or_low={or_low}")

    return or_high, or_low, data_date

def _looks_like_missing_columns(df: pd.DataFrame, required: tuple[str, ...]) -> str | None:
    missing = [c for c in required if c not in getattr(df, "columns", [])]
    if missing:
        return ",".join(missing)
    return None


def _classify_intraday_exception(exc: Exception | str) -> tuple[str, str]:
    """Map real exceptions/messages into explicit, stable failure buckets."""
    msg = str(exc or "").strip()
    low = msg.lower()

    exact_codes = {
        "daily_empty",
        "avg20_calc_missing",
        "avg20_below_threshold",
        "daily_ok",
        "intraday_empty",
        "intraday_session_filter_empty",
        "intraday_missing_columns",
        "intraday_or_window",
        "intraday_or_window_missing_open",
        "intraday_or_window_missing_close",
        "intraday_or_window_incomplete",
        "intraday_invalid_or_values",
        "filtered_or_range",
        "filtered_today_dollar_vol",
        "filtered_rvol",
        "filtered_risk_per_share",
        "filtered_shares",
        "filtered_notional",
        "filtered_no_valid_plan",
        "filtered_price",
        "filtered_avg20_dollar_vol",
        "no_recent_touch",
        "risk_out_of_bounds",
        "position_sizing_invalid",
        "rr_sigma_vwap_invalid",
        "rr_not_enough_rth_bars",
    }
    if msg in exact_codes:
        return msg, msg

    normalized_msg = msg
    for prefix in ("RuntimeError: ", "ValueError: ", "Exception: "):
        if normalized_msg.startswith(prefix):
            normalized_msg = normalized_msg[len(prefix):].strip()
            break

    if normalized_msg in exact_codes:
        return normalized_msg, msg

    if normalized_msg.startswith("daily_missing_columns:"):
        return "daily_missing_columns", msg
    if normalized_msg.startswith("intraday_missing_columns:"):
        return "intraday_missing_columns", msg
    if normalized_msg.startswith("intraday_or_window_missing_open:"):
        return "intraday_or_window_missing_open", msg
    if normalized_msg.startswith("intraday_or_window_incomplete:"):
        return "intraday_or_window_incomplete", msg
    if normalized_msg.startswith("intraday_invalid_or_values:"):
        return "intraday_invalid_or_values", msg

    if "provider returned empty intraday bars" in low:
        return "intraday_empty", msg
    if "not enough rth bars for rr window" in low:
        return "rr_not_enough_rth_bars", msg
    if "sigma/vwap invalid" in low:
        return "rr_sigma_vwap_invalid", msg

    if "timed out" in low or "timeout" in low:
        return "bars_fetch_error_timeout", msg
    if "429" in low or "rate limit" in low or "too many requests" in low:
        return "bars_fetch_error_rate_limit", msg
    if "401" in low or "403" in low or "unauthorized" in low or "forbidden" in low:
        return "bars_fetch_error_auth", msg
    if "400" in low or "bad request" in low or "invalid symbol" in low:
        return "bars_fetch_error_http_400", msg
    if "404" in low or "not found" in low:
        return "bars_fetch_error_http_404", msg
    if "connection" in low or "socket" in low or "dns" in low or "ssl" in low or "network" in low:
        return "bars_fetch_error_network", msg
    if "parse" in low or "json" in low or "decode" in low or "schema" in low:
        return "bars_fetch_error_parse", msg
    if "alpaca get_bars_range failed" in low or "get_stock_bars" in low:
        return "bars_fetch_error_provider", msg

    return "plan_build_exception", msg


def _append_failure_sample(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    code: str,
    item: dict[str, Any],
    limit_per_code: int = 5,
) -> None:
    """Keep a small forensic sample per real failure code without fake placeholders."""
    if not code:
        code = "unknown"
    bucket = buckets.setdefault(str(code), [])
    symbol = str(item.get("symbol") or "")
    stage = str(item.get("stage") or "")
    error = str(item.get("error") or "")
    for existing in bucket:
        if (
            str(existing.get("symbol") or "") == symbol
            and str(existing.get("stage") or "") == stage
            and str(existing.get("error") or "") == error
        ):
            return
    if len(bucket) < int(limit_per_code):
        bucket.append(dict(item))


def _build_failure_samples_by_code(
    items: list[dict[str, Any]],
    *,
    limit_per_code: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items or []:
        raw_code = item.get("code")
        if not raw_code:
            raw_code, _ = _classify_intraday_exception(item.get("error") or "")
        normalized = dict(item)
        normalized["code"] = raw_code
        _append_failure_sample(buckets, code=str(raw_code), item=normalized, limit_per_code=limit_per_code)
    return buckets


def _prefilter_daily(provider: AlpacaProvider, symbols: list[str], cfg: ORBConfig) -> tuple[list[str], dict[str, int], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Daily prefilter so we don't fetch intraday for thousands of symbols.
    This is NOT simulated data; it's real daily OHLCV from the active market-data provider.
    """
    counts = {
        "normalized_skipped": 0,
        "daily_empty": 0,
        "filtered_price": 0,
        "filtered_avg20_dollar_vol": 0,
        "avg20_calc_missing": 0,
        "avg20_below_threshold": 0,
        "daily_ok": 0,
    }
    errors: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    thresholds_used = {
        "min_price": float(cfg.min_price),
        "max_price": float(cfg.max_price),
        "min_avg20_dollar_vol": float(cfg.min_avg20_dollar_vol),
        "prefilter_period": str(cfg.prefilter_period),
    }

    # Normalize + skip unsupported symbols first
    normalized: list[str] = []
    for s in symbols:
        y = to_provider_symbol(s)
        if y is None:
            counts["normalized_skipped"] += 1
            continue

        if "." in y:
            counts["normalized_skipped"] += 1
            continue

        normalized.append(y)

    # Batch download daily
    daily_map = provider.download_daily_batch(normalized, period=cfg.prefilter_period)

    shortlisted: list[str] = []
    for sym in normalized:
        df = daily_map.get(sym)
        if df is None or df.empty:
            counts["daily_empty"] += 1
            err_item = {"symbol": sym, "stage": "daily", "error": "daily_empty", "code": "daily_empty"}
            errors.append(err_item)
            if len(samples) < 25:
                samples.append({
                    "symbol": sym,
                    "last_close": None,
                    "avg20_volume": None,
                    "avg20_dollar_vol": None,
                    "threshold_min_avg20_dollar_vol": float(cfg.min_avg20_dollar_vol),
                    "passed": False,
                    "reason": "daily_empty",
                })
            continue

        try:
            missing_cols = _looks_like_missing_columns(df, ("Close", "Volume"))
            if missing_cols:
                raise RuntimeError(f"daily_missing_columns:{missing_cols}")

            last_close = float(df["Close"].astype(float).iloc[-1])
            avg20_vol = avg_daily_volume(df, window=20)
            avg20_dollar_vol = (float(avg20_vol) * last_close) if (avg20_vol is not None and avg20_vol > 0) else None

            sample = {
                "symbol": sym,
                "last_close": round(last_close, 6),
                "avg20_volume": None if avg20_vol is None else round(float(avg20_vol), 6),
                "avg20_dollar_vol": None if avg20_dollar_vol is None else round(float(avg20_dollar_vol), 6),
                "threshold_min_avg20_dollar_vol": float(cfg.min_avg20_dollar_vol),
                "passed": False,
                "reason": None,
            }

            if not (cfg.min_price <= last_close <= cfg.max_price):
                counts["filtered_price"] += 1
                sample["reason"] = "filtered_price"
                if len(samples) < 25:
                    samples.append(sample)
                continue

            if avg20_dollar_vol is None or avg20_dollar_vol <= 0:
                counts["filtered_avg20_dollar_vol"] += 1
                counts["avg20_calc_missing"] += 1
                sample["reason"] = "avg20_calc_missing"
                if len(samples) < 25:
                    samples.append(sample)
                continue

            if avg20_dollar_vol < cfg.min_avg20_dollar_vol:
                counts["filtered_avg20_dollar_vol"] += 1
                counts["avg20_below_threshold"] += 1
                sample["reason"] = "avg20_below_threshold"
                if len(samples) < 25:
                    samples.append(sample)
                continue

            counts["daily_ok"] += 1
            shortlisted.append(sym)
            sample["passed"] = True
            sample["reason"] = "daily_ok"
            if len(samples) < 25:
                samples.append(sample)
        except Exception as e:
            code, detail = _classify_intraday_exception(e)
            errors.append({"symbol": sym, "stage": "daily", "error": detail, "code": code})
            if len(samples) < 25:
                samples.append({
                    "symbol": sym,
                    "last_close": None,
                    "avg20_volume": None,
                    "avg20_dollar_vol": None,
                    "threshold_min_avg20_dollar_vol": float(cfg.min_avg20_dollar_vol),
                    "passed": False,
                    "reason": code,
                    "error": detail,
                })

    return shortlisted, counts, errors, samples, thresholds_used

def build_orb_plan(provider: AlpacaProvider, symbol: str, cfg: ORBConfig, *, session_date: date, exec_style: str = 'retest') -> Candidate:
    """Build ORB plan for a symbol for a specific US equity session date.

    Tenants:
      - Uses real provider data only.
      - If provider returns empty for the session, raises a real exception (recorded by caller).
    """
    # Prefer explicit date range fetch so we align to a concrete session.
    def _normalize_intraday_session(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        x = df.sort_index().copy()
        try:
            if getattr(x.index, "tz", None) is None:
                x.index = x.index.tz_localize("UTC")
            x = x.tz_convert(ET)
        except Exception as e:
            raise RuntimeError("intraday_timezone_convert_failed") from e

        x = x.loc[x.index.date == session_date]
        if x.empty:
            return x

        x = x.between_time("09:30", "16:00", inclusive="both")
        return x

    intraday = None
    range_err = None

    try:
        if hasattr(provider, "get_bars_range"):
            intraday = provider.get_bars_range(
                symbol=symbol,
                interval="1m",
                from_d=session_date,
                to_d=session_date,
                include_prepost=False,
            )
    except Exception as e:
        range_err = e

    try:
        intraday_session = _normalize_intraday_session(intraday)
    except RuntimeError:
        raise
    except Exception as e:
        code, _ = _classify_intraday_exception(e)
        raise RuntimeError(code) from e

    def _has_opening_print(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        day = session_date
        nominal_open = datetime.combine(day, dtime(9, 30), tzinfo=ET)
        opening_band = df.loc[
            (df.index >= nominal_open) &
            (df.index <= nominal_open + pd.Timedelta(minutes=10))
        ]
        return not opening_band.empty

    if intraday_session.empty or not _has_opening_print(intraday_session):
        try:
            fallback = provider.get_bars(
                BarsRequest(
                    symbol=symbol,
                    interval="1m",
                    period="5d",
                    include_prepost=False,
                )
            )
        except Exception as e:
            code, _ = _classify_intraday_exception(e)
            raise RuntimeError(code) from e

        intraday_session = _normalize_intraday_session(fallback)

    if intraday_session.empty:
        if range_err is not None:
            code, _ = _classify_intraday_exception(range_err)
            raise RuntimeError(code)
        raise RuntimeError("intraday_session_filter_empty")

    intraday = intraday_session

    if intraday.empty:
        raise RuntimeError("intraday_session_filter_empty")

    missing_intraday_cols = _looks_like_missing_columns(intraday, ("Open", "High", "Low", "Close", "Volume"))
    if missing_intraday_cols:
        raise RuntimeError(f"intraday_missing_columns:{missing_intraday_cols}")

    last_price = float(intraday["Close"].astype(float).iloc[-1])
    if not (cfg.min_price <= last_price <= cfg.max_price):
        raise ValueError("filtered_price")

    first_close = float(intraday["Close"].astype(float).iloc[0])
    pct_change = ((last_price - first_close) / first_close * 100.0) if first_close > 0 else None

    soft_fail_reasons: list[str] = []

    or_high, or_low, data_date = _get_opening_range_5m(intraday)
    or_mid = (or_high + or_low) / 2.0
    or_range_pct = (or_high - or_low) / or_mid * 100.0 if or_mid > 0 else 999.0
    if not (cfg.min_or_range_pct <= or_range_pct <= cfg.max_or_range_pct):
        soft_fail_reasons.append("filtered_or_range")

    try:
        vw = vwap(intraday)
        vwap_last = float(vw.iloc[-1])
        above_vwap = bool(last_price > vwap_last)
        tr = trend_state_1m(intraday, vw=vw, lookback=15)
        trend_state = str(tr.get("state"))
        trend_slope_pct = tr.get("slope_pct_lookback")
        vwap_delta_pct = tr.get("vwap_delta_pct")
    except Exception:
        above_vwap = None
        vwap_last = None
        vwap_delta_pct = None
        trend_state = None
        trend_slope_pct = None

    # Daily history for avg20 dollar vol and RVOL
    daily = provider.get_daily_history(symbol, period=cfg.prefilter_period).sort_index()
    avg20_vol = avg_daily_volume(daily, window=20)
    if avg20_vol is None or avg20_vol <= 0:
        raise ValueError("filtered_avg20_dollar_vol")
    avg20_dollar_vol = float(avg20_vol) * float(daily["Close"].astype(float).iloc[-1])
    if avg20_dollar_vol < cfg.min_avg20_dollar_vol:
        raise ValueError("filtered_avg20_dollar_vol")

    today_vol = float(intraday["Volume"].astype(float).sum())
    today_dollar_vol = today_vol * last_price
    if today_dollar_vol < cfg.min_today_dollar_vol:
        soft_fail_reasons.append("filtered_today_dollar_vol")

    rvol = today_vol / float(avg20_vol)
    if rvol < cfg.min_rvol:
        soft_fail_reasons.append("filtered_rvol")

    daily_ctx = daily.copy()
    daily_ctx["day"] = daily_ctx.index.date
    prev_rows = daily_ctx[daily_ctx["day"] < session_date]

    prev_close = float(prev_rows["Close"].astype(float).iloc[-1]) if not prev_rows.empty else None
    prev_prev_close = float(prev_rows["Close"].astype(float).iloc[-2]) if len(prev_rows) >= 2 else None
    prev_day_high = float(prev_rows["High"].astype(float).iloc[-1]) if (not prev_rows.empty and "High" in prev_rows.columns) else None
    recent_20d_high = float(prev_rows["High"].astype(float).tail(20).max()) if (not prev_rows.empty and "High" in prev_rows.columns) else None

    dist_from_prev_close_pct = ((last_price - prev_close) / prev_close * 100.0) if (prev_close is not None and prev_close > 0) else None
    prev_day_change_pct = ((prev_close - prev_prev_close) / prev_prev_close * 100.0) if (prev_close is not None and prev_prev_close is not None and prev_prev_close > 0) else None
    dist_above_prev_day_high_pct = ((last_price - prev_day_high) / prev_day_high * 100.0) if (prev_day_high is not None and prev_day_high > 0) else None
    dist_above_20d_high_pct = ((last_price - recent_20d_high) / recent_20d_high * 100.0) if (recent_20d_high is not None and recent_20d_high > 0) else None

    # Dual-side ORB plans (breakout + breakdown)
    buf_h = _buffer(or_high, cfg)
    buf_l = _buffer(or_low, cfg)

    # Long
    long_entry = or_high + buf_h
    long_stop = or_low - buf_l
    long_risk = long_entry - long_stop
    long_seed_valid = bool(long_risk > 0)
    long_valid = long_seed_valid and (cfg.min_risk_per_share <= long_risk <= cfg.max_risk_per_share)
    long_shares = None
    long_notional = None
    long_2r = None
    long_3r = None
    if long_seed_valid:
        s = int(math.floor(cfg.risk_dollars / max(long_risk, 1e-9))) if long_risk > 0 else 0
        n = s * long_entry
        long_shares = max(0, s)
        long_notional = n
        long_2r = long_entry + 2.0 * long_risk
        long_3r = long_entry + 3.0 * long_risk
        if not (long_valid and s >= cfg.min_shares and n <= cfg.max_notional):
            long_valid = False

    # Short
    short_entry = or_low - buf_l
    short_stop = or_high + buf_h
    short_risk = short_stop - short_entry
    short_seed_valid = bool(short_risk > 0)
    short_valid = short_seed_valid and (cfg.min_risk_per_share <= short_risk <= cfg.max_risk_per_share)
    short_shares = None
    short_notional = None
    short_2r = None
    short_3r = None
    if short_seed_valid:
        s = int(math.floor(cfg.risk_dollars / max(short_risk, 1e-9))) if short_risk > 0 else 0
        n = s * short_entry
        short_shares = max(0, s)
        short_notional = n
        short_2r = short_entry - 2.0 * short_risk
        short_3r = short_entry - 3.0 * short_risk
        if not (short_valid and s >= cfg.min_shares and n <= cfg.max_notional):
            short_valid = False

    if not long_seed_valid and not short_seed_valid:
        raise ValueError("filtered_no_valid_plan")
    if not long_valid and not short_valid:
        soft_fail_reasons.append("filtered_no_valid_plan")

    # Choose best side (simple heuristic: prefer side with larger RVOL alignment)
    # If both valid, prefer above-VWAP for long, below-VWAP for short; otherwise prefer tighter risk.
    if long_seed_valid and not short_seed_valid:
        best_side = "long"
    elif short_seed_valid and not long_seed_valid:
        best_side = "short"
    else:
        # both seed-valid
        if above_vwap is True:
            best_side = "long"
        elif above_vwap is False:
            best_side = "short"
        else:
            best_side = "long" if long_risk <= short_risk else "short"

    if best_side == "long":
        entry = float(long_entry)
        stop = float(long_stop)
        risk_per_share = float(long_risk)
        shares = int(long_shares or 0)
        notional = float(long_notional or 0.0)
        target_2r = float(long_2r or (entry + 2.0 * risk_per_share))
        target_3r = float(long_3r or (entry + 3.0 * risk_per_share))
    else:
        entry = float(short_entry)
        stop = float(short_stop)
        risk_per_share = float(short_risk)
        shares = int(short_shares or 0)
        notional = float(short_notional or 0.0)
        target_2r = float(short_2r or (entry - 2.0 * risk_per_share))
        target_3r = float(short_3r or (entry - 3.0 * risk_per_share))

    notes = (
        f"5m ORB | date {data_date} | OR% {or_range_pct:.2f} | $Vol {today_dollar_vol/1e6:.1f}M | RVOL {rvol:.2f} | best {best_side}"
    )
    if soft_fail_reasons:
        notes = f"{notes} | soft {'/'.join(soft_fail_reasons)}"


    long_triggered = bool(long_seed_valid and (last_price >= float(long_entry)))
    short_triggered = bool(short_seed_valid and (last_price <= float(short_entry)))
    long_trigger_delta = float(long_entry - last_price) if long_seed_valid else None
    short_trigger_delta = float(last_price - short_entry) if short_seed_valid else None

    def _orb_retest_snapshot(
        df: pd.DataFrame,
        *,
        side: str,
        entry_px: float,
        stop_px: float,
        last_px: float,
    ) -> tuple[bool, float | None, str | None]:
        if df is None or df.empty:
            return False, None, None

        x = df.sort_index().copy()
        risk = max(abs(float(entry_px) - float(stop_px)), 1e-9)
        touch_band = 0.10 * risk
        confirm_band = 0.20 * risk

        if side == "long":
            triggered = x[x["High"].astype(float) >= float(entry_px)]
            if triggered.empty:
                return False, None, None
            first_trigger_ts = triggered.index[0]
            post = x.loc[x.index >= first_trigger_ts].copy()
            touches = post[
                (post["Low"].astype(float) <= float(entry_px) + touch_band) &
                (post["High"].astype(float) >= float(entry_px) - touch_band)
            ]
            if touches.empty:
                return False, None, None
            touch_ts = touches.index[0]
            distance_r = (float(last_px) - float(entry_px)) / risk
            ready = bool(
                float(last_px) >= float(entry_px) and
                abs(distance_r) <= float(confirm_band / risk)
            )
            return ready, float(distance_r), touch_ts.isoformat()

        if side == "short":
            triggered = x[x["Low"].astype(float) <= float(entry_px)]
            if triggered.empty:
                return False, None, None
            first_trigger_ts = triggered.index[0]
            post = x.loc[x.index >= first_trigger_ts].copy()
            touches = post[
                (post["High"].astype(float) >= float(entry_px) - touch_band) &
                (post["Low"].astype(float) <= float(entry_px) + touch_band)
            ]
            if touches.empty:
                return False, None, None
            touch_ts = touches.index[0]
            distance_r = (float(entry_px) - float(last_px)) / risk
            ready = bool(
                float(last_px) <= float(entry_px) and
                abs(distance_r) <= float(confirm_band / risk)
            )
            return ready, float(distance_r), touch_ts.isoformat()

        return False, None, None

    long_retest_ready, long_retest_distance_r, long_retest_touch_ts = (
        _orb_retest_snapshot(
            intraday,
            side="long",
            entry_px=float(long_entry),
            stop_px=float(long_stop),
            last_px=float(last_price),
        )
        if long_seed_valid else (False, None, None)
    )

    short_retest_ready, short_retest_distance_r, short_retest_touch_ts = (
        _orb_retest_snapshot(
            intraday,
            side="short",
            entry_px=float(short_entry),
            stop_px=float(short_stop),
            last_px=float(last_price),
        )
        if short_seed_valid else (False, None, None)
    )

    orb_retest_ready = long_retest_ready if best_side == "long" else short_retest_ready
    orb_retest_distance_r = long_retest_distance_r if best_side == "long" else short_retest_distance_r
    orb_retest_touch_ts = long_retest_touch_ts if best_side == "long" else short_retest_touch_ts

    return Candidate(
        symbol=symbol,
        data_date=data_date,
        last_price=last_price,
        pct_change=pct_change,
        rvol=rvol,
        today_dollar_vol=today_dollar_vol,
        avg20_dollar_vol=avg20_dollar_vol,
        or_high=or_high,
        or_low=or_low,
        or_range_pct=or_range_pct,
        above_vwap=above_vwap,
        vwap_last=vwap_last,
        vwap_delta_pct=vwap_delta_pct,
        trend_state=trend_state,
        trend_slope_pct=trend_slope_pct,
        best_side=best_side,
        entry=entry,
        stop=stop,
        scan_date=session_date.isoformat(),
        stop_loss=stop,
        take_profit=target_2r,
        target_2r=target_2r,
        target_3r=target_3r,
        risk_per_share=risk_per_share,
        shares=shares,
        notional=notional,
        long_entry=float(long_entry) if long_seed_valid else None,
        long_stop=float(long_stop) if long_seed_valid else None,
        long_2r=float(long_2r) if long_seed_valid and long_2r is not None else None,
        long_3r=float(long_3r) if long_seed_valid and long_3r is not None else None,
        long_risk_per_share=float(long_risk) if long_seed_valid else None,
        long_shares=int(long_shares) if long_seed_valid and long_shares is not None else None,
        long_notional=float(long_notional) if long_seed_valid and long_notional is not None else None,
        short_entry=float(short_entry) if short_seed_valid else None,
        short_stop=float(short_stop) if short_seed_valid else None,
        short_2r=float(short_2r) if short_seed_valid and short_2r is not None else None,
        short_3r=float(short_3r) if short_seed_valid and short_3r is not None else None,
        short_risk_per_share=float(short_risk) if short_seed_valid else None,
        short_shares=int(short_shares) if short_seed_valid and short_shares is not None else None,
        short_notional=float(short_notional) if short_seed_valid and short_notional is not None else None,
        long_triggered=long_triggered if long_seed_valid else None,
        short_triggered=short_triggered if short_seed_valid else None,
        long_trigger_delta=long_trigger_delta,
        short_trigger_delta=short_trigger_delta,
        notes=notes,
        exec_style=str(exec_style),
        orb_retest_ready=bool(orb_retest_ready),
        orb_retest_distance_r=orb_retest_distance_r,
        orb_retest_touch_ts=orb_retest_touch_ts,
        monitor_seed=True,
        monitor_seed_reasons=soft_fail_reasons or None,
        tradable_now=bool(long_valid or short_valid),
        prev_close=prev_close,
        prev_day_high=prev_day_high,
        recent_20d_high=recent_20d_high,
        dist_from_prev_close_pct=dist_from_prev_close_pct,
        prev_day_change_pct=prev_day_change_pct,
        dist_above_prev_day_high_pct=dist_above_prev_day_high_pct,
        dist_above_20d_high_pct=dist_above_20d_high_pct,
    )

def _passes_orb_live_execution_gate(
    c: Candidate,
    *,
    exec_style: str = "retest",
    min_minutes_after_open: int = 10,
    breakout_now_min_ml_score: float = 0.45,
    breakout_now_max_chase_r: float = 0.10,
    retest_min_ml_score: float = 0.40,
    retest_max_chase_r: float = 0.20,
    retest_max_distance_r: float = 0.20,
) -> tuple[bool, str | None]:
    phase = _session_phase_et()
    now_et = datetime.now(tz=ET)
    minutes_since_open = max(0, (now_et.hour * 60 + now_et.minute) - (9 * 60 + 30))

    side = (c.best_side or '').lower()
    entry = float(c.entry or 0.0)
    stop = float(c.stop or 0.0)
    price = float(c.last_price or 0.0)
    ml_score = float(c.ml_score or 0.0) if c.ml_score is not None else 0.0

    risk = max(abs(entry - stop), 1e-9)
    if side == "long":
        chase_r = max(0.0, (price - entry) / risk) if entry > 0 and price > 0 else 999.0
    elif side == "short":
        chase_r = max(0.0, (entry - price) / risk) if entry > 0 and price > 0 else 999.0
    else:
        return False, "orb_no_side"

    if phase in {"overnight", "premarket"}:
        return False, "orb_not_live_before_open"

    if phase == "open" and minutes_since_open < int(min_minutes_after_open):
        return False, "orb_wait_after_open"

    mode = (exec_style or "retest").strip().lower()

    if mode == "breakout_now":
        if phase != "open":
            return False, "orb_breakout_phase_blocked"
        if ml_score < float(breakout_now_min_ml_score):
            return False, "orb_breakout_ml_too_low"
        if chase_r > float(breakout_now_max_chase_r):
            return False, "orb_breakout_too_extended"
        if side == "long" and not bool(c.long_triggered):
            return False, "orb_breakout_not_triggered"
        if side == "short" and not bool(c.short_triggered):
            return False, "orb_breakout_not_triggered"
        return True, None

    if ml_score < float(retest_min_ml_score):
        return False, "orb_retest_ml_too_low"
    if chase_r > float(retest_max_chase_r):
        return False, "orb_retest_too_extended"
    if not bool(c.orb_retest_ready):
        return False, "orb_retest_not_confirmed"
    if c.orb_retest_distance_r is None or abs(float(c.orb_retest_distance_r)) > float(retest_max_distance_r):
        return False, "orb_retest_not_near_entry"

    return True, None

def _is_market_open_et(now_et: datetime) -> bool:
    # 9:30-16:00 ET regular session (we treat close as inclusive-ish)
    t = now_et.time()
    return (t >= dtime(9, 30)) and (t <= dtime(16, 0))


def resolve_session_date(provider, probe_symbol: str = "AAPL") -> date:
    """
    Resolve the trading session date to use for scans/plans.

    Rules:
    - On weekends, use the most recent Friday.
    - On weekdays before 09:30 ET, use the most recent completed weekday session.
    - On/after 09:30 ET, prefer today.
    - Only fall back when today is clearly not tradable (holiday / no bars at all).
    """
    now_et = datetime.now(_ET)
    today_et = now_et.date()

    def _prior_weekday(d: date) -> date:
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    # Weekend -> most recent Friday
    if today_et.weekday() >= 5:
        d = today_et
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    # Before the regular session opens, ORB should use the last completed session.
    if now_et.time() < dtime(9, 30):
        return _prior_weekday(today_et)

    try:
        intraday = provider.get_bars(
            BarsRequest(
                symbol=probe_symbol,
                interval="1m",
                period="1d",
                include_prepost=True,
            )
        )

        if intraday is not None and not intraday.empty:
            idx = intraday.index
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize("UTC").tz_convert(_ET)
            else:
                idx = idx.tz_convert(_ET)

            # If any returned bar belongs to today ET, use today.
            if (idx.date == today_et).any():
                return today_et

            # After the open on a weekday, prefer today even if bars are sparse.
            return today_et

    except Exception:
        # Provider hiccup after the open on a weekday: prefer today rather than poisoning the scan with stale dates.
        return today_et

    # No bars at all after the open on a weekday -> likely holiday; walk back to prior weekday.
    return _prior_weekday(today_et)



def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _session_phase_et(now: datetime | None = None) -> str:
    n = now or datetime.now(tz=ET)
    hhmm = n.hour * 100 + n.minute
    if hhmm < 400:
        return "overnight"
    if hhmm < 930:
        return "premarket"
    if hhmm < 1030:
        return "open"
    if hhmm < 1400:
        return "midday"
    if hhmm < 1600:
        return "powerhour"
    if hhmm < 2000:
        return "afterhours"
    return "overnight"


def _pick_regime_profile(name: str | None, candidates: list[Candidate]) -> dict:
    requested = (name or "auto").strip().lower()
    phase = _session_phase_et()
    if requested in {"auto", ""}:
        if phase in {"premarket", "afterhours", "overnight"}:
            selected = "extended"
        elif phase == "open":
            selected = "open_drive"
        elif phase == "midday":
            selected = "midday"
        else:
            selected = "powerhour"
    else:
        selected = requested
    avg_rvol = (sum(float(c.rvol or 0.0) for c in candidates) / len(candidates)) if candidates else 0.0
    avg_or = (sum(float(c.or_range_pct or 0.0) for c in candidates) / len(candidates)) if candidates else 0.0
    profiles = {
        "extended": {
            "score_weights": {"ml": 1.0, "sentiment": 0.6, "catalyst": 1.1, "rule": 0.15},
            "confidence_weights": {"setup": 0.35, "tape": 0.15, "alignment": 0.20, "catalyst": 0.30},
            "bias": 0.0,
        },
        "open_drive": {
            "score_weights": {"ml": 1.0, "sentiment": 0.7, "catalyst": 0.7, "rule": 0.20},
            "confidence_weights": {"setup": 0.40, "tape": 0.25, "alignment": 0.25, "catalyst": 0.10},
            "bias": 0.03 if avg_rvol >= 2.0 else 0.0,
        },
        "midday": {
            "score_weights": {"ml": 1.0, "sentiment": 0.8, "catalyst": 0.8, "rule": 0.10},
            "confidence_weights": {"setup": 0.30, "tape": 0.20, "alignment": 0.25, "catalyst": 0.25},
            "bias": -0.02 if avg_or < 0.9 else 0.0,
        },
        "powerhour": {
            "score_weights": {"ml": 1.0, "sentiment": 0.7, "catalyst": 0.9, "rule": 0.15},
            "confidence_weights": {"setup": 0.35, "tape": 0.20, "alignment": 0.25, "catalyst": 0.20},
            "bias": 0.01,
        },
    }
    prof = dict(profiles.get(selected, profiles["open_drive"]))
    prof["selected"] = selected
    prof["phase"] = phase
    prof["avg_rvol"] = round(avg_rvol, 3)
    prof["avg_or_range_pct"] = round(avg_or, 3)
    return prof


def _candidate_setup_rule_score(c: Candidate) -> float:
    side = (c.best_side or '').lower()
    rvol_score = _clip(float(c.rvol or 0.0) / 3.0, 0.0, 1.0)
    vol_score = _clip(float(c.today_dollar_vol or 0.0) / 25_000_000.0, 0.0, 1.0)
    or_score = _clip(1.0 - abs(float(c.or_range_pct or 0.0) - 1.8) / 2.0, 0.0, 1.0)

    trend = (c.trend_state or '').lower()
    vwap_delta = float(c.vwap_delta_pct or 0.0)

    if side == 'long':
        directional_score = 1.0 if trend in {'up', 'reclaim_vwap'} and vwap_delta > 0 else 0.0
        trigger_score = 1.0 if c.long_triggered else _clip(
            1.0 - abs(float(c.long_trigger_delta or 0.0)) / max(float(c.entry or 1.0), 1e-6) / 0.02,
            0.0,
            1.0,
        )
    else:
        directional_score = 1.0 if trend in {'down', 'lost_vwap'} and vwap_delta < 0 else 0.0
        trigger_score = 1.0 if c.short_triggered else _clip(
            1.0 - abs(float(c.short_trigger_delta or 0.0)) / max(float(c.entry or 1.0), 1e-6) / 0.02,
            0.0,
            1.0,
        )

    return float(round(
        0.25 * rvol_score +
        0.25 * vol_score +
        0.15 * or_score +
        0.20 * directional_score +
        0.15 * trigger_score,
        6
    ))


def _alignment_score(c: Candidate) -> float:
    trend = (c.trend_state or '').lower()
    side = (c.best_side or '').lower()
    vwap_good = (side == 'long' and c.above_vwap is True) or (side == 'short' and c.above_vwap is False)
    trend_good = (side == 'long' and trend in {'up', 'reclaim_vwap'}) or (side == 'short' and trend in {'down', 'lost_vwap'})
    trend_neutral = trend in {'chop', ''}
    score = (0.55 if vwap_good else 0.15)
    if trend_good:
        score += 0.35
    elif trend_neutral:
        score += 0.15
    else:
        score += 0.05
    if (side == 'long' and c.long_triggered) or (side == 'short' and c.short_triggered):
        score += 0.10
    return _clip(score, 0.0, 1.0)


def _tape_proxy_score(c: Candidate) -> float:
    side = (c.best_side or '').lower()
    slope = float(c.trend_slope_pct or 0.0)
    vwap_delta = float(c.vwap_delta_pct or 0.0)
    trend = (c.trend_state or '').lower()

    if side == 'long':
        slope_score = _clip((slope + 0.10) / 1.20, 0.0, 1.0)
        vwap_score = _clip((vwap_delta + 0.20) / 2.00, 0.0, 1.0)
        state_score = {
            'up': 1.00,
            'reclaim_vwap': 0.85,
            'chop': 0.25,
            'lost_vwap': 0.05,
            'down': 0.00,
            '': 0.20,
        }.get(trend, 0.20)
    else:
        slope_score = _clip(((-slope) + 0.10) / 1.20, 0.0, 1.0)
        vwap_score = _clip(((-vwap_delta) + 0.20) / 2.00, 0.0, 1.0)
        state_score = {
            'down': 1.00,
            'lost_vwap': 0.85,
            'chop': 0.25,
            'reclaim_vwap': 0.05,
            'up': 0.00,
            '': 0.20,
        }.get(trend, 0.20)

    return float(round(
        0.45 * slope_score +
        0.35 * vwap_score +
        0.20 * state_score,
        6
    ))


def _rr_setup_rule_score(c: Candidate, feats: dict[str, float]) -> float:
    rvol_score = _clip(float(c.rvol or 0.0) / 2.5, 0.0, 1.0)
    vol_score = _clip(float(c.today_dollar_vol or 0.0) / 20_000_000.0, 0.0, 1.0)
    or_score = _clip(1.0 - abs(float(c.or_range_pct or 0.0) - 2.0) / 2.5, 0.0, 1.0)
    touch_age = float(getattr(c, 'rr_touch_age_min', feats.get('touch_age_min', 999.0)) or 999.0)
    touch_score = _clip(1.0 - (touch_age / 45.0), 0.0, 1.0)
    return float(round(
        0.25 * rvol_score +
        0.25 * vol_score +
        0.20 * or_score +
        0.30 * touch_score,
        6
    ))


def _rr_alignment_score(c: Candidate) -> float:
    trend = (c.trend_state or '').lower()
    vwap_delta = float(c.vwap_delta_pct or 0.0)
    above_vwap = bool(c.above_vwap) if c.above_vwap is not None else False
    score = 0.0
    if above_vwap:
        score += 0.40
    elif vwap_delta >= -0.50:
        score += 0.30
    elif vwap_delta >= -1.00:
        score += 0.20
    else:
        score += 0.10

    if trend == 'reclaim_vwap':
        score += 0.35
    elif trend in {'chop', 'lost_vwap'}:
        score += 0.22
    elif trend == 'up':
        score += 0.20
    else:
        score += 0.08

    return _clip(score, 0.0, 1.0)


def _rr_tape_proxy_score(c: Candidate, feats: dict[str, float]) -> float:
    slope = float(feats.get('slope', c.trend_slope_pct or 0.0))
    vwap_delta = float(c.vwap_delta_pct or 0.0)
    trend = (c.trend_state or '').lower()
    slope_score = _clip((slope + 0.05) / 0.50, 0.0, 1.0)
    vwap_score = _clip((vwap_delta + 1.20) / 2.40, 0.0, 1.0)
    state_score = {
        'reclaim_vwap': 1.00,
        'chop': 0.65,
        'lost_vwap': 0.55,
        'up': 0.70,
        'down': 0.20,
        '': 0.45,
    }.get(trend, 0.45)
    return float(round(
        0.35 * slope_score +
        0.40 * vwap_score +
        0.25 * state_score,
        6
    ))


def _passes_orb_directional_gate(
    c: Candidate,
    *,
    long_only: bool = True,
    min_pct_over_vwap: float = 1.0,
    allow_reclaim: bool = True,
    reject_chop: bool = True,
    min_ml_score: float = 0.35,
    max_chase_r: float = 0.35,
) -> tuple[bool, str | None]:
    side = (c.best_side or '').lower()
    trend = (c.trend_state or '').lower()
    vwap_delta = float(c.vwap_delta_pct or 0.0)
    above_vwap = bool(c.above_vwap) if c.above_vwap is not None else False

    entry = float(c.entry or 0.0)
    stop = float(c.stop or 0.0)
    price = float(c.last_price or 0.0)
    risk = max(abs(entry - stop), 1e-9)

    if c.ml_score is not None and float(c.ml_score) < float(min_ml_score):
        return False, 'ml_too_low'

    if long_only and side != 'long':
        return False, 'not_long'

    if side == 'long':
        chase_r = max(0.0, (price - entry) / risk) if entry > 0 and price > 0 else 0.0
        if chase_r > float(max_chase_r):
            return False, 'chase_too_high'
        if not above_vwap:
            if not (allow_reclaim and trend == 'reclaim_vwap' and vwap_delta >= 0.0):
                return False, 'below_vwap'
        if reject_chop and trend == 'chop':
            return False, 'trend_chop'
        if trend in {'down', 'lost_vwap'}:
            return False, 'trend_against'
        if vwap_delta < float(min_pct_over_vwap):
            return False, 'below_min_pct_over_vwap'

    if side == 'short':
        chase_r = max(0.0, (entry - price) / risk) if entry > 0 and price > 0 else 0.0
        if chase_r > float(max_chase_r):
            return False, 'chase_too_high'
        if above_vwap:
            if not (allow_reclaim and trend == 'lost_vwap' and vwap_delta <= 0.0):
                return False, 'above_vwap'
        if reject_chop and trend == 'chop':
            return False, 'trend_chop'
        if trend in {'up', 'reclaim_vwap'}:
            return False, 'trend_against'
        if vwap_delta > -float(min_pct_over_vwap):
            return False, 'above_max_pct_under_vwap'

    return True, None


def _passes_final_orb_rank_gate(c: Candidate, *, min_combined: float = 0.55, min_grade: str = 'B') -> tuple[bool, str | None]:
    grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    if float(c.combined_score or 0.0) < float(min_combined):
        return False, 'combined_too_low'
    if float(c.confidence_score or 0.0) < 72.0:
        return False, 'confidence_too_low'
    if grade_order.get(c.confidence_grade or 'F', 0) < grade_order.get(str(min_grade).upper(), 3):
        return False, 'grade_too_low'
    return True, None


def _orb_chase_gate_reason(c: Candidate, *, max_chase_r: float = 0.35) -> str | None:
    side = (c.best_side or '').lower()
    entry = float(c.entry or 0.0)
    stop = float(c.stop or 0.0)
    price = float(c.last_price or 0.0)
    risk = max(abs(entry - stop), 1e-9)

    if side == 'long':
        chase_r = max(0.0, (price - entry) / risk) if entry > 0 and price > 0 else 0.0
    elif side == 'short':
        chase_r = max(0.0, (entry - price) / risk) if entry > 0 and price > 0 else 0.0
    else:
        return 'orb_no_side'

    if chase_r > float(max_chase_r):
        return 'chase_too_high'
    return None


def _passes_orb_discovery_gate(
    c: Candidate,
    *,
    long_only: bool = True,
    min_pct_over_vwap: float = 1.0,
    allow_reclaim: bool = True,
    reject_chop: bool = True,
    min_ml_score: float = 0.35,
    min_combined: float = 0.55,
    min_grade: str = 'B',
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok, reason = _passes_orb_directional_gate(
        c,
        long_only=long_only,
        min_pct_over_vwap=min_pct_over_vwap,
        allow_reclaim=allow_reclaim,
        reject_chop=reject_chop,
        min_ml_score=min_ml_score,
        max_chase_r=999999.0,
    )
    if not ok and reason:
        reasons.append(reason)

    ok2, reason2 = _passes_final_orb_rank_gate(
        c,
        min_combined=min_combined,
        min_grade=min_grade,
    )
    if not ok2 and reason2:
        reasons.append(reason2)

    return (not reasons), reasons


def _passes_orb_trade_ready_gate(
    c: Candidate,
    *,
    exec_style: str = "retest",
    min_minutes_after_open: int = 10,
    breakout_now_min_ml_score: float = 0.45,
    breakout_now_max_chase_r: float = 0.10,
    retest_min_ml_score: float = 0.40,
    retest_max_chase_r: float = 0.20,
    directional_max_chase_r: float = 0.35,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    chase_reason = _orb_chase_gate_reason(c, max_chase_r=directional_max_chase_r)
    if chase_reason:
        reasons.append(chase_reason)

    ok_exec, exec_reason = _passes_orb_live_execution_gate(
        c,
        exec_style=exec_style,
        min_minutes_after_open=min_minutes_after_open,
        breakout_now_min_ml_score=breakout_now_min_ml_score,
        breakout_now_max_chase_r=breakout_now_max_chase_r,
        retest_min_ml_score=retest_min_ml_score,
        retest_max_chase_r=retest_max_chase_r,
    )
    if not ok_exec and exec_reason:
        reasons.append(exec_reason)

    return (not reasons), list(dict.fromkeys(reasons))


def _rr_reclaim_quality(c: Candidate, feats: dict[str, float]) -> float:
    trend = (c.trend_state or '').lower()
    vwap_delta = float(c.vwap_delta_pct or 0.0)
    slope = float(feats.get('slope', 0.0))
    relvol15 = float(feats.get('relvol15', 0.0))
    trend_score = {
        'reclaim_vwap': 1.00,
        'up': 0.80,
        'chop': 0.25,
        'lost_vwap': 0.05,
        'down': 0.00,
        '': 0.20,
    }.get(trend, 0.20)
    return float(round(
        0.40 * _clip((vwap_delta + 2.0) / 3.0, 0.0, 1.0) +
        0.30 * _clip((slope + 0.05) / 0.50, 0.0, 1.0) +
        0.20 * _clip(relvol15 / 2.0, 0.0, 1.0) +
        0.10 * trend_score,
        6
    ))


def _rr_rule_score(c: Candidate, feats: dict[str, float]) -> float:
    z = abs(float(feats.get('zscore', 0.0)))
    relvol5 = float(feats.get('relvol5', 0.0))
    relvol15 = float(feats.get('relvol15', 0.0))
    slope = float(feats.get('slope', 0.0))
    dist_entry = float(feats.get('dist_to_entry_sig', 0.0))
    trend = (c.trend_state or '').lower()
    vwap_delta = float(c.vwap_delta_pct or 0.0)

    z_score = _clip((z - 1.2) / 1.8, 0.0, 1.0)
    vol_score = _clip((relvol5 + relvol15) / 4.0, 0.0, 1.0)
    reclaim_score = _clip((vwap_delta + 2.0) / 3.0, 0.0, 1.0)
    slope_score = _clip((slope + 0.05) / 0.50, 0.0, 1.0)
    proximity_score = _clip(1.0 - dist_entry / 1.5, 0.0, 1.0)
    trend_score = {
        'reclaim_vwap': 1.00,
        'up': 0.80,
        'chop': 0.25,
        'lost_vwap': 0.05,
        'down': 0.00,
        '': 0.20,
    }.get(trend, 0.20)

    return float(round(
        0.20 * z_score +
        0.20 * vol_score +
        0.20 * reclaim_score +
        0.15 * slope_score +
        0.15 * proximity_score +
        0.10 * trend_score,
        6
    ))


def _passes_rr_actionable_long_gate(c: Candidate, feats: dict[str, float]) -> tuple[bool, str | None, float, float]:
    trend = (c.trend_state or '').lower()
    vwap_delta = float(c.vwap_delta_pct or 0.0)
    slope = float(feats.get('slope', 0.0))
    price = float(c.last_price or 0.0)
    entry = float(c.entry or 0.0)
    stop = float(c.stop or 0.0)

    risk = max(entry - stop, 1e-6)
    chase_r = (price - entry) / risk if price > 0 and entry > 0 else 999.0
    touch_age_min = max(0.0, 390.0 - float(feats.get('tod_min', 390.0)))

    # Deliberately looser RR actionable gate: reclaims often begin from messy,
    # slightly-below-VWAP conditions. Reject only clearly broken states.
    if trend == 'down' and not (vwap_delta >= -0.40 and slope >= 0.03):
        return False, 'rr_trend_still_down', touch_age_min, chase_r
    if trend == 'lost_vwap' and not (vwap_delta >= -0.75 and slope >= -0.02):
        return False, 'rr_trend_still_down', touch_age_min, chase_r
    if trend == 'chop' and not (vwap_delta >= -0.35 and slope >= -0.03):
        return False, 'rr_chop', touch_age_min, chase_r
    if vwap_delta < -4.0:
        return False, 'rr_too_far_below_vwap', touch_age_min, chase_r
    if chase_r > 2.0:
        return False, 'rr_chase_too_high', touch_age_min, chase_r
    if float(feats.get('tod_min', 0.0)) >= 360.0:
        return False, 'rr_too_late', touch_age_min, chase_r
    return True, None, touch_age_min, chase_r


def _grade_for_confidence(score: float) -> str:
    if score >= 85:
        return 'A+'
    if score >= 78:
        return 'A'
    if score >= 70:
        return 'B'
    if score >= 62:
        return 'C'
    if score >= 50:
        return 'D'
    return 'F'


def _orb_base_score(c: Candidate) -> float:
    s = 0.0
    s += min(30.0, (c.today_dollar_vol or 0.0) / 25_000_000.0 * 30.0)
    s += min(30.0, (c.rvol or 0.0) / 8.0 * 30.0)
    if c.above_vwap is True:
        s += 10.0
    s += max(0.0, 20.0 - abs(c.or_range_pct - 2.5) * 6.0)
    dist = (c.entry - c.last_price) / c.entry * 100.0
    s += 10.0 if dist <= 1.5 else (5.0 if dist <= 3.0 else 0.0)
    return s


def _rank_candidates_for_optional_enrichment(candidates: list[Candidate], *, use_ml: bool) -> list[Candidate]:
    if use_ml:
        return sorted(
            candidates,
            key=lambda x: (float(x.ml_score or 0.0), float(x.rvol or 0.0), float(x.today_dollar_vol or 0.0), _orb_base_score(x)),
            reverse=True,
        )
    return sorted(candidates, key=_orb_base_score, reverse=True)


def _fetch_orb_sentiment_map(*, candidates: list[Candidate], use_ml: bool, limit: int, sentiment_provider: str, data_failures: list[dict[str, str]], env_int_func) -> dict[str, float]:
    sentiment_map: dict[str, float] = {}
    if not candidates:
        return sentiment_map

    pre_sorted = _rank_candidates_for_optional_enrichment(candidates, use_ml=use_ml)
    topn_default = max(50, int(limit) * 3)
    topn = env_int_func('ORB_SENTIMENT_TOPN', topn_default)
    target_syms = [c.symbol for c in pre_sorted[: min(topn, len(pre_sorted))]]

    try:
        svc = SentimentService(provider=sentiment_provider)
    except Exception as e:
        data_failures.append({'symbol': '__all__', 'stage': 'sentiment_init', 'error': str(e)})
        return sentiment_map

    from concurrent.futures import ThreadPoolExecutor, as_completed

    sentiment_workers = env_int_func('ORB_SENTIMENT_WORKERS', 8)

    def _sent_one(sym: str) -> tuple[str, float | None, str | None]:
        try:
            b = svc.fetch(sym, limit=6)
            return sym, float(b.score), None
        except Exception as e:
            return sym, None, str(e)

    with ThreadPoolExecutor(max_workers=sentiment_workers) as ex:
        futs = [ex.submit(_sent_one, s) for s in target_syms]
        for fut in as_completed(futs):
            sym, score, err = fut.result()
            if score is not None:
                sentiment_map[sym] = score
            else:
                data_failures.append({'symbol': sym, 'stage': 'sentiment', 'error': err or 'unknown'})

    return sentiment_map


def _fetch_orb_catalyst_map(*, candidates: list[Candidate], use_ml: bool, limit: int, provider, catalyst_topn: int | None, catalyst_lookback_hours: int, data_failures: list[dict[str, str]], env_int_func) -> dict[str, Any]:
    catalyst_map: dict[str, Any] = {}
    if not candidates:
        return catalyst_map

    pre_sorted = _rank_candidates_for_optional_enrichment(candidates, use_ml=use_ml)
    topn_default = max(25, int(limit) * 2)
    ctopn = int(catalyst_topn or env_int_func('ORB_CATALYST_TOPN', topn_default))
    target_syms = [c.symbol for c in pre_sorted[: min(ctopn, len(pre_sorted))]]

    try:
        csvc = CatalystService(provider=provider)
        catalyst_map = csvc.fetch_batch(target_syms, per_symbol_limit=6, lookback_hours=int(catalyst_lookback_hours))
        for sym, bundle in catalyst_map.items():
            if getattr(bundle, 'error', None):
                data_failures.append({'symbol': sym, 'stage': 'catalyst', 'error': str(bundle.error)})
    except Exception as e:
        data_failures.append({'symbol': '__all__', 'stage': 'catalyst_init', 'error': f'{type(e).__name__}: {e}'})
        catalyst_map = {}

    return catalyst_map


def _apply_orb_enrichment_to_candidate(c: Candidate, *, sentiment_map: dict[str, float], catalyst_map: dict[str, Any]) -> None:
    c.sentiment_score = sentiment_map.get(c.symbol)
    cb = catalyst_map.get(c.symbol) if isinstance(catalyst_map, dict) else None
    if cb is not None:
        c.catalyst_score = float(getattr(cb, 'score', 0.0) or 0.0)
        c.catalyst_confidence = float(getattr(cb, 'confidence', 0.0) or 0.0)
        c.catalyst_strength = float(getattr(cb, 'strength', 0.0) or 0.0)
        c.catalyst_article_count = int(getattr(cb, 'article_count', 0) or 0)
        c.catalyst_freshness_hours = getattr(cb, 'freshness_hours', None)
        c.catalyst_tags = list(getattr(cb, 'tags', []) or [])
    else:
        c.catalyst_score = None
        c.catalyst_confidence = None
        c.catalyst_strength = None
        c.catalyst_article_count = None
        c.catalyst_freshness_hours = None
        c.catalyst_tags = None


def _score_orb_candidate(c: Candidate, *, use_ml: bool, sentiment_alpha: float, catalyst_alpha: float, regime_selected: str | None) -> dict[str, float | str | None]:
    ml_base = float(c.ml_score or 0.0) if use_ml else 0.0
    rule_norm = _candidate_setup_rule_score(c)
    tape_q = _tape_proxy_score(c)
    align_q = _alignment_score(c)
    sent_term = float(sentiment_alpha) * float(c.sentiment_score or 0.0) if c.sentiment_score is not None else 0.0
    cat_term = float(catalyst_alpha) * float(c.catalyst_score or 0.0) if c.catalyst_score is not None else 0.0
    score = (
        0.38 * ml_base +
        0.22 * rule_norm +
        0.18 * tape_q +
        0.14 * align_q +
        0.04 * sent_term +
        0.04 * cat_term
    )
    c.regime_profile = str(regime_selected)
    c.regime_adjustment = float(round(score - ml_base, 6))
    c.combined_score = float(round(score, 6))

    cat_q = _clip(float(c.catalyst_confidence or 0.0) / 100.0, 0.0, 1.0)
    conf = 100.0 * (
        0.30 * rule_norm +
        0.30 * tape_q +
        0.25 * align_q +
        0.15 * cat_q
    )
    if c.sentiment_score is not None and c.catalyst_score is not None:
        if (c.sentiment_score >= 0 and c.catalyst_score >= 0) or (c.sentiment_score <= 0 and c.catalyst_score <= 0):
            conf += 3.0
        else:
            conf -= 3.0
    if (c.best_side == 'long' and c.long_triggered) or (c.best_side == 'short' and c.short_triggered):
        conf += 2.0
    conf = _clip(conf, 0.0, 100.0)
    c.confidence_score = float(round(conf, 2))
    c.confidence_grade = _grade_for_confidence(conf)

    return {
        'ml_base': round(ml_base, 6),
        'rule_norm': round(rule_norm, 6),
        'tape_q': round(tape_q, 6),
        'align_q': round(align_q, 6),
        'sent_term': round(sent_term, 6),
        'cat_term': round(cat_term, 6),
        'combined_score': round(float(c.combined_score or 0.0), 6),
        'confidence_score': round(float(c.confidence_score or 0.0), 2),
    }


def _apply_orb_gate_state(c: Candidate, *, score_parts: dict[str, float | str | None], exec_style_requested: str, exec_style: str, long_only_flag: bool, min_vwap_enabled: bool, min_vwap_value: float, no_chop_enabled: bool, use_ml: bool, orb_min_ml_value: float, min_combined_enabled: bool, min_combined_value: float, min_grade_enabled: bool, min_grade_value: str, orb_min_minutes_after_open_value: int, orb_breakout_now_min_ml_value: float, orb_breakout_now_max_chase_r_value: float, orb_retest_min_ml_value: float, orb_retest_max_chase_r_value: float, orb_max_chase_r_value: float) -> tuple[bool, list[str], bool, list[str]]:
    discovery_ok, discovery_reasons = _passes_orb_discovery_gate(
        c,
        long_only=long_only_flag,
        min_pct_over_vwap=min_vwap_value if min_vwap_enabled else -999.0,
        allow_reclaim=True,
        reject_chop=no_chop_enabled,
        min_ml_score=orb_min_ml_value if use_ml else 0.0,
        min_combined=min_combined_value if min_combined_enabled else 0.0,
        min_grade=min_grade_value if min_grade_enabled else 'F',
    )
    trade_ready_ok, trade_ready_reasons = _passes_orb_trade_ready_gate(
        c,
        exec_style=exec_style,
        min_minutes_after_open=orb_min_minutes_after_open_value,
        breakout_now_min_ml_score=orb_breakout_now_min_ml_value,
        breakout_now_max_chase_r=orb_breakout_now_max_chase_r_value,
        retest_min_ml_score=orb_retest_min_ml_value,
        retest_max_chase_r=orb_retest_max_chase_r_value,
        directional_max_chase_r=orb_max_chase_r_value,
    )

    c.discovery_passes = bool(discovery_ok)
    c.discovery_fail_reasons = discovery_reasons or None
    c.trade_ready_passes = bool(discovery_ok and trade_ready_ok)
    c.trade_ready_fail_reasons = trade_ready_reasons or None
    c.gate_passes = bool(discovery_ok and trade_ready_ok)
    c.gate_fail_reasons = list(dict.fromkeys([*(discovery_reasons or []), *(trade_ready_reasons or [])]))
    c.score_breakdown = {
        'strategy': 'orb',
        'exec_style_requested': exec_style_requested,
        'exec_style_effective': exec_style,
        'discovery_gate_passes': bool(discovery_ok),
        'discovery_gate_reasons': list(discovery_reasons or []),
        'trade_ready_passes': bool(discovery_ok and trade_ready_ok),
        'trade_ready_reasons': list(trade_ready_reasons or []),
        'execution_gate_passes': bool(trade_ready_ok),
        'execution_gate_reason': ((trade_ready_reasons or [None])[0] if trade_ready_reasons else None),
        'ml_score': score_parts['ml_base'],
        'rule_score': score_parts['rule_norm'],
        'tape_score': score_parts['tape_q'],
        'alignment_score': score_parts['align_q'],
        'sentiment_term': score_parts['sent_term'],
        'catalyst_term': score_parts['cat_term'],
        'combined_score': score_parts['combined_score'],
        'confidence_score': score_parts['confidence_score'],
        'confidence_grade': c.confidence_grade,
        'gate_passes': c.gate_passes,
    }
    return bool(discovery_ok), list(discovery_reasons or []), bool(trade_ready_ok), list(trade_ready_reasons or [])


def _orb_seed_sort_key(x: Candidate):
    penalties = 0.0
    reasons = set(x.monitor_seed_reasons or [])
    penalties += 0.12 * float('filtered_rvol' in reasons)
    penalties += 0.10 * float('filtered_or_range' in reasons)
    penalties += 0.08 * float('filtered_today_dollar_vol' in reasons)
    penalties += 0.12 * float('filtered_no_valid_plan' in reasons)
    penalties += 0.10 * float('ml_too_low' in reasons)
    penalties += 0.08 * float('confidence_too_low' in reasons)
    tradable_bonus = 0.20 if bool(x.tradable_now) else 0.0
    return (
        float(x.combined_score or 0.0) - penalties + tradable_bonus,
        float(x.confidence_score or 0.0),
        float(x.ml_score or 0.0),
        float(x.rvol or 0.0),
        float(x.today_dollar_vol or 0.0),
        _orb_base_score(x),
    )


def _orb_build_failure_payload(*, sym: str, err: Exception, session_date: date, strategy: str, exec_style: str) -> dict[str, Any]:
    raw_code = f"{type(err).__name__}: {err}"
    code, detail = _classify_intraday_exception(raw_code)
    failure = PlanBuildFailure(
        code=code,
        message=detail,
        stage="intraday",
        symbol=sym,
        context={
            "session_date": session_date.isoformat(),
            "interval": "1m",
            "strategy": strategy,
            "exec_style": exec_style,
        },
        cause_type=type(err).__name__,
    )
    return failure.to_dict()


def _orb_fail_item_from_payload(payload: dict[str, Any], *, sym: str, session_date: date, strategy: str, exec_style: str) -> dict[str, Any]:
    code = str(payload.get("code") or "other_data_failures")
    detail = str(payload.get("message") or payload.get("error") or code)
    fail_item = {
        "symbol": sym,
        "stage": str(payload.get("stage") or "intraday"),
        "error": detail,
        "code": code,
        "session_date": session_date.isoformat(),
        "interval": "1m",
        "strategy": strategy,
        "exec_style": exec_style,
    }
    ctx = payload.get("context")
    if isinstance(ctx, dict):
        for k, v in ctx.items():
            if k not in fail_item and v is not None:
                fail_item[k] = v
    cause_type = payload.get("cause_type")
    if cause_type:
        fail_item["cause_type"] = cause_type
    return fail_item


def _orb_fail_item_from_payload(payload: dict[str, Any], *, sym: str, session_date: date, strategy: str, exec_style: str) -> dict[str, Any]:
    code = str(payload.get("code") or "other_data_failures")
    detail = str(payload.get("message") or payload.get("error") or code)
    fail_item = {
        "symbol": sym,
        "stage": str(payload.get("stage") or "intraday"),
        "error": detail,
        "code": code,
        "session_date": session_date.isoformat(),
        "interval": "1m",
        "strategy": strategy,
        "exec_style": exec_style,
    }
    ctx = payload.get("context")
    if isinstance(ctx, dict):
        for k, v in ctx.items():
            if k not in fail_item and v is not None:
                fail_item[k] = v
    cause_type = payload.get("cause_type")
    if cause_type:
        fail_item["cause_type"] = cause_type
    return fail_item


def scan_symbols(
    symbols: List[str],
    cfg: ORBConfig,
    limit: int = 25,
    **kwargs,
) -> Dict[str, Any]:
    """
    Scan symbols and produce ORB candidates.

    Tenants:
      - NO fake/sample data; uses Alpaca market data (real provider) only.
      - On provider failures, records real failures and continues where appropriate.
      - Every method/import exists; full file; no placeholders.
    """
    # Strategy dispatch (default ORB). Range Reversion uses its own plan+ML.
    exec_style_requested = str(kwargs.get('exec_style') or kwargs.get('execution') or 'retest').strip().lower() or 'retest'
    strategy = str(kwargs.get('strategy') or kwargs.get('scan_strategy') or 'orb').strip().lower() or 'orb'
    if strategy in {'range_reversion','rr','range'}:
        return scan_range_reversion_symbols(symbols, cfg, limit=limit, **kwargs)

    stream_cache = kwargs.get("stream_cache")
    streaming_only = bool(kwargs.get("streaming_only", False) or kwargs.get("runtime_streaming_only", False))
    if stream_cache is not None and streaming_only:
        return _scan_symbols_streaming_only(
            symbols,
            cfg,
            limit=limit,
            stream_cache=stream_cache,
            strategy=strategy,
            exec_style=str(exec_style_requested),
            use_ml=bool(kwargs.get("use_ml", False)),
        )

    phase_now = _session_phase_et()
    exec_style = exec_style_requested
    if strategy == 'orb' and exec_style_requested == 'breakout_now' and phase_now != 'open':
        exec_style = 'retest'

    import os
    # HARD LOCK: Alpaca-only market data (tenant-safe, no silent fallback)
    mdp = (os.getenv('ORB_MARKET_DATA_PROVIDER') or 'alpaca').strip().lower()
    if mdp != 'alpaca':
        raise RuntimeError(f"Scan disabled: ORB_MARKET_DATA_PROVIDER={mdp} (must be alpaca)")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Backward/forward compatibility: allow callers to pass feature flags as kwargs
    use_ml: bool = bool(kwargs.get("use_ml", False))
    use_sentiment: bool = bool(kwargs.get("use_sentiment", False))
    sentiment_provider: str = str(kwargs.get("sentiment_provider", "auto"))
    sentiment_alpha: float = float(kwargs.get("sentiment_alpha", 0.15))
    use_catalyst: bool = bool(kwargs.get("use_catalyst", str(os.getenv("ORB_USE_CATALYST", "0")).lower() in {"1", "true", "yes", "on"}))
    catalyst_alpha: float = float(kwargs.get("catalyst_alpha", os.getenv("ORB_CATALYST_ALPHA", 0.08)))
    catalyst_topn = kwargs.get("catalyst_topn", None)
    catalyst_lookback_hours: int = int(kwargs.get("catalyst_lookback_hours", os.getenv("ORB_CATALYST_LOOKBACK_HOURS", 72)))
    regime_profile: str = str(kwargs.get("regime_profile", os.getenv("ORB_REGIME_PROFILE", "auto"))).strip().lower() or "auto"

    # ---------- Provider + lightweight caching (huge speed win) ----------
    #
    # build_orb_plan() and ORBRanker both call provider.get_bars()/get_daily_history().
    # Cache at the provider boundary so ML scoring reuses already-fetched data.
    base_provider = kwargs.get('provider') or AlpacaProvider()

    class _CachedProvider:
        def __init__(self, inner: AlpacaProvider):
            self._p = inner
            self.name = getattr(inner, "name", "alpaca")

            self._lock = threading.Lock()
            self._bars_cache: dict[tuple, pd.DataFrame] = {}
            self._daily_cache: dict[tuple, pd.DataFrame] = {}
            self._daily_batch_cache: dict[tuple, dict[str, pd.DataFrame]] = {}

        def get_bars(self, req: BarsRequest, timeout_s: int | None = None) -> pd.DataFrame:
            key = (req.symbol, req.interval, req.period, bool(getattr(req, "include_prepost", False)))
            with self._lock:
                hit = self._bars_cache.get(key)
            if hit is not None:
                return hit
            # Pass through timeout_s when supported by the underlying provider.
            try:
                df = self._p.get_bars(req, timeout_s=timeout_s)
            except TypeError:
                df = self._p.get_bars(req)
            # cache a copy reference; DataFrames are immutable for our usage (we sort/copy when needed)
            with self._lock:
                self._bars_cache[key] = df
            return df

        def get_daily_history(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
            key = (symbol, period)
            with self._lock:
                hit = self._daily_cache.get(key)
            if hit is not None:
                return hit
            df = self._p.get_daily_history(symbol, period=period)
            with self._lock:
                self._daily_cache[key] = df
            return df

        def download_daily_batch(self, symbols: list[str], period: str = "1mo") -> dict[str, pd.DataFrame]:
            # Symbols list order doesn't matter for caching
            key = (tuple(sorted(symbols)), period)
            with self._lock:
                hit = self._daily_batch_cache.get(key)
            if hit is not None:
                return hit
            m = self._p.download_daily_batch(symbols, period=period)

            # Populate per-symbol daily cache so later get_daily_history() can reuse the batch fetch
            with self._lock:
                for sym, df in (m or {}).items():
                    self._daily_cache[(sym, period)] = df
            with self._lock:
                self._daily_batch_cache[key] = m
            return m

        # Some downstream components (session resolver, ML feature builders) need range fetching.
        # Proxy through if the underlying provider supports it.
        def get_bars_range(self, *args, **kwargs):
            if hasattr(self._p, "get_bars_range"):
                return self._p.get_bars_range(*args, **kwargs)
            raise AttributeError("Underlying provider has no get_bars_range")

        # Forward any other attributes/methods to the underlying provider to avoid accidental breakage.
        def __getattr__(self, item):
            return getattr(self._p, item)

    provider = _CachedProvider(base_provider)

    session_date = resolve_session_date(base_provider)

    # ---------- Daily prefilter (batch) ----------
    shortlisted, pre_counts, daily_errors, prefilter_samples, thresholds_used = _prefilter_daily(base_provider, symbols, cfg)

    candidates: List[Candidate] = []

    # Categorized rejections/failures
    reject_counts = {
        "filtered_or_range": 0,
        "filtered_today_dollar_vol": 0,
        "filtered_rvol": 0,
        "filtered_risk_per_share": 0,
        "filtered_shares": 0,
        "filtered_notional": 0,
        "filtered_no_valid_plan": 0,
        "intraday_empty": 0,
        "intraday_session_filter_empty": 0,
        "intraday_timezone_convert_failed": 0,
        "intraday_missing_columns": 0,
        "intraday_or_window": 0,
        "intraday_or_window_missing_open": 0,
        "intraday_or_window_missing_close": 0,
        "intraday_or_window_incomplete": 0,
        "intraday_invalid_or_values": 0,
        "bars_fetch_error_timeout": 0,
        "bars_fetch_error_rate_limit": 0,
        "bars_fetch_error_auth": 0,
        "bars_fetch_error_http_400": 0,
        "bars_fetch_error_http_404": 0,
        "bars_fetch_error_network": 0,
        "bars_fetch_error_parse": 0,
        "bars_fetch_error_provider": 0,
        "plan_build_exception": 0,
        "ml_too_low": 0,
        "chase_too_high": 0,
        "orb_not_live_before_open": 0,
        "orb_wait_after_open": 0,
        "orb_breakout_phase_blocked": 0,
        "orb_breakout_ml_too_low": 0,
        "orb_breakout_too_extended": 0,
        "orb_breakout_not_triggered": 0,
        "orb_retest_ml_too_low": 0,
        "orb_retest_too_extended": 0,
        "orb_retest_not_confirmed": 0,
        "orb_retest_not_near_entry": 0,
        "other_data_failures": 0,
    }
    data_failures: List[Dict[str, str]] = daily_errors[:]  # real failures only
    failure_samples_by_code: dict[str, list[dict[str, str]]] = {}

    # ---------- Intraday builds (parallel; network-bound) ----------
    #
    # Tunables (all optional):
    #   ORB_SCAN_WORKERS: threads used for build_orb_plan()
    #   ORB_SENTIMENT_WORKERS: threads used for sentiment calls
    #   ORB_SENTIMENT_TOPN: only compute sentiment for top-N candidates (post-ML)
    #
    def _env_int(name: str, default: int) -> int:
        try:
            v = int(os.environ.get(name, str(default)).strip())
            return v if v > 0 else default
        except Exception:
            return default

    default_workers = min(16, max(4, (os.cpu_count() or 8) * 2))
    scan_workers = _env_int("ORB_SCAN_WORKERS", default_workers)

    # Alpaca rate-limits aggressively; clamp parallelism for reliability (override by ORB_SCAN_WORKERS_MAX).
    scan_workers_max = _env_int("ORB_SCAN_WORKERS_MAX", 16)
    scan_workers = max(1, min(scan_workers, scan_workers_max))

    def _build_one(sym: str) -> tuple[str, Candidate | None, dict[str, Any] | None]:
        try:
            return sym, build_orb_plan(provider, sym, cfg, session_date=session_date, exec_style=exec_style), None
        except Exception as e:
            return sym, None, _orb_build_failure_payload(
                sym=sym,
                err=e,
                session_date=session_date,
                strategy=strategy,
                exec_style=exec_style,
            )

    # Important: keep deterministic-ish order by submitting in symbol order, but we collect as completed.
    with ThreadPoolExecutor(max_workers=scan_workers) as ex:
        futs = [ex.submit(_build_one, sym) for sym in shortlisted]
        for fut in as_completed(futs):
            sym, c, err = fut.result()
            if c is not None:
                candidates.append(c)
                continue

            # rejection/failure mapping
            if isinstance(err, dict):
                code = str(err.get("code") or "other_data_failures")
                fail_item = _orb_fail_item_from_payload(
                    err,
                    sym=sym,
                    session_date=session_date,
                    strategy=strategy,
                    exec_style=exec_style,
                )
                detail = str(fail_item.get("error") or code)
            else:
                raw_code = err or "unknown"
                code, detail = _classify_intraday_exception(raw_code)
                fail_item = {
                    "symbol": sym,
                    "stage": "intraday",
                    "error": detail,
                    "code": code,
                    "session_date": session_date.isoformat(),
                    "interval": "1m",
                    "strategy": strategy,
                    "exec_style": exec_style,
                }

            if code in reject_counts:
                reject_counts[code] += 1
            elif code.startswith("filtered_"):
                reject_counts[code] = reject_counts.get(code, 0) + 1
            else:
                reject_counts["other_data_failures"] += 1

            data_failures.append(fail_item)
            _append_failure_sample(failure_samples_by_code, code=code, item=fail_item)

    # ---------- Optional ML ranking (strict A/B runtime models only) ----------
    ml_scores: Dict[str, float] = {}
    ml_bucket: Dict[str, str] = {}
    if use_ml and candidates:
        ml_out = score_orb_candidates(candidates, provider=provider)
        ml_scores = dict(ml_out.get("scores") or {})
        ml_bucket = dict(ml_out.get("bucket_by_symbol") or {})
        data_failures.extend(list(ml_out.get("failures") or []))

    for c in candidates:
        c.ml_score = ml_scores.get(c.symbol)
        c.model_bucket = ml_bucket.get(c.symbol)
        # provisional combined (before sentiment)
        c.combined_score = float(c.ml_score or 0.0)

    # ---------- Optional sentiment boost (compute only for top-N for speed) ----------
    sentiment_map: Dict[str, float] = {}
    if use_sentiment and candidates:
        sentiment_map = _fetch_orb_sentiment_map(
            candidates=candidates,
            use_ml=use_ml,
            limit=limit,
            sentiment_provider=sentiment_provider,
            data_failures=data_failures,
            env_int_func=_env_int,
        )

    # ---------- Optional catalyst/news score (compute only for top-N for speed) ----------
    catalyst_map = {}
    if use_catalyst and candidates:
        catalyst_map = _fetch_orb_catalyst_map(
            candidates=candidates,
            use_ml=use_ml,
            limit=limit,
            provider=provider,
            catalyst_topn=catalyst_topn,
            catalyst_lookback_hours=int(catalyst_lookback_hours),
            data_failures=data_failures,
            env_int_func=_env_int,
        )

    # Regime profile used for ranking/confidence (based on ET clock + real scan feature mix)
    regime = _pick_regime_profile(regime_profile, candidates)

    long_only_flag = bool(kwargs.get("long_only", True))
    min_grade_enabled = bool(kwargs.get("min_grade_enabled", True))
    min_grade_value = str(kwargs.get("min_grade", "B")).strip().upper() or "B"
    min_combined_enabled = bool(kwargs.get("min_combined_enabled", True))
    min_combined_value = float(kwargs.get("min_combined_score", 0.40))
    no_chop_enabled = bool(kwargs.get("no_chop_enabled", True))
    min_vwap_enabled = bool(kwargs.get("min_vwap_enabled", True))
    min_vwap_value = float(kwargs.get("min_pct_over_vwap", 1.0))
    orb_min_minutes_after_open_value = int(kwargs.get("orb_min_minutes_after_open", 10))
    orb_breakout_now_min_ml_value = float(kwargs.get("orb_breakout_now_min_ml_score", 0.45))
    orb_breakout_now_max_chase_r_value = float(kwargs.get("orb_breakout_now_max_chase_r", 0.10))
    orb_retest_min_ml_value = float(kwargs.get("orb_retest_min_ml_score", 0.40))
    orb_retest_max_chase_r_value = float(kwargs.get("orb_retest_max_chase_r", 0.20))
    orb_min_ml_value = float(kwargs.get("orb_min_ml_score", 0.35))
    orb_max_chase_r_value = float(kwargs.get("orb_max_chase_r", 0.35))

    monitor_first_mode = bool(kwargs.get("monitor_first_mode", True))
    admissible: list[Candidate] = []
    discovery_candidates: list[Candidate] = []
    seed_candidates: list[Candidate] = []
    rejected_candidates: list[Candidate] = []

    def _seedable_monitor_candidate(c: Candidate, reasons: list[str], execution_reason: str | None) -> bool:
        if not monitor_first_mode:
            return False
        if long_only_flag and str(c.best_side or "").lower() != "long":
            return False
        if float(c.last_price or 0.0) <= 0.0:
            return False
        if float(c.entry or 0.0) <= 0.0 or float(c.stop or 0.0) <= 0.0:
            return False
        hard_blockers = {"not_long"}
        if any(r in hard_blockers for r in (reasons or [])):
            return False
        c.monitor_seed = True
        merged = list(dict.fromkeys([*(c.monitor_seed_reasons or []), *(reasons or []), *([execution_reason] if execution_reason else [])]))
        c.monitor_seed_reasons = merged or None
        c.tradable_now = bool(not reasons and not execution_reason)
        return True

    # Apply sentiment + catalyst + final combined score
    for c in candidates:
        _apply_orb_enrichment_to_candidate(c, sentiment_map=sentiment_map, catalyst_map=catalyst_map)
        score_parts = _score_orb_candidate(
            c,
            use_ml=use_ml,
            sentiment_alpha=float(sentiment_alpha),
            catalyst_alpha=float(catalyst_alpha),
            regime_selected=regime.get('selected'),
        )
        discovery_ok, discovery_reasons, trade_ready_ok, trade_ready_reasons = _apply_orb_gate_state(
            c,
            score_parts=score_parts,
            exec_style_requested=exec_style_requested,
            exec_style=exec_style,
            long_only_flag=long_only_flag,
            min_vwap_enabled=min_vwap_enabled,
            min_vwap_value=min_vwap_value,
            no_chop_enabled=no_chop_enabled,
            use_ml=use_ml,
            orb_min_ml_value=orb_min_ml_value,
            min_combined_enabled=min_combined_enabled,
            min_combined_value=min_combined_value,
            min_grade_enabled=min_grade_enabled,
            min_grade_value=min_grade_value,
            orb_min_minutes_after_open_value=orb_min_minutes_after_open_value,
            orb_breakout_now_min_ml_value=orb_breakout_now_min_ml_value,
            orb_breakout_now_max_chase_r_value=orb_breakout_now_max_chase_r_value,
            orb_retest_min_ml_value=orb_retest_min_ml_value,
            orb_retest_max_chase_r_value=orb_retest_max_chase_r_value,
            orb_max_chase_r_value=orb_max_chase_r_value,
        )

        if c.catalyst_article_count:
            cat_txt = f"Cat {float(c.catalyst_score or 0.0):+.2f}/{int(c.catalyst_article_count)}"
            c.notes = f"{c.notes} | {cat_txt}" if c.notes else cat_txt

        if c.discovery_passes:
            discovery_candidates.append(c)

        if c.gate_passes:
            c.monitor_seed = True
            c.tradable_now = True
            admissible.append(c)
            seed_candidates.append(c)
        else:
            for _reason in (c.gate_fail_reasons or []):
                reject_counts[_reason] = reject_counts.get(_reason, 0) + 1
            if _seedable_monitor_candidate(c, c.gate_fail_reasons or [], (trade_ready_reasons or [None])[0] if trade_ready_reasons else None):
                seed_candidates.append(c)
            rejected_candidates.append(c)

    if use_ml or use_sentiment or use_catalyst:
        admissible.sort(key=_orb_seed_sort_key, reverse=True)
        seed_candidates.sort(key=_orb_seed_sort_key, reverse=True)
    else:
        admissible.sort(key=_orb_base_score, reverse=True)
        seed_candidates.sort(key=_orb_base_score, reverse=True)

    # Best trader setup: the main surfaced candidate pool should remain strictly
    # trade-ready/admissible. Monitor seeds are preserved in separate counters and
    # rejected/debug payloads for watchlist/monitor workflows, but do not pollute
    # the primary trading list.
    top_pool = admissible
    top = top_pool[: int(limit)]
    catalyst_enriched = sum(1 for c in candidates if c.catalyst_article_count)

    if not prefilter_samples:
        for sym in shortlisted[:10]:
            prefilter_samples.append({
                "symbol": sym,
                "passed": True,
                "reason": "daily_ok",
            })

    if not failure_samples_by_code:
        failure_samples_by_code = _build_failure_samples_by_code(data_failures, limit_per_code=5)
    else:
        for _code, _items in _build_failure_samples_by_code(data_failures, limit_per_code=5).items():
            bucket = failure_samples_by_code.setdefault(_code, [])
            for _it in _items:
                if len(bucket) < 5:
                    bucket.append(_it)

    return {
        "provider": provider.name,
        "scan_date": session_date.isoformat(),
        "regime": regime,
        "debug": {
            "session_date_used": session_date.isoformat(),
            "failure_samples": data_failures[:10],
            "failure_samples_by_code": failure_samples_by_code,
            "catalyst_enriched": catalyst_enriched,
        },
        "count": len(top),
        "candidates_total": len(top_pool),
        "discovery_total": len(discovery_candidates),
        "seed_candidates_total": len(seed_candidates),
        "tradable_now_total": len(admissible),
        "trade_ready_total": len(admissible),
        "rejected_total": len(rejected_candidates),
        "candidates": [({**asdict(c), "price": float(c.last_price or 0.0), "scan_date": session_date.isoformat(), "scan_ts": (c.scan_ts or datetime.now(timezone.utc).isoformat())}) for c in top],
        "seed_candidates": [({**asdict(c), "price": float(c.last_price or 0.0), "scan_date": session_date.isoformat(), "scan_ts": (c.scan_ts or datetime.now(timezone.utc).isoformat())}) for c in seed_candidates[: int(limit)]],
        "rejected_candidates": [({**asdict(c), "price": float(c.last_price or 0.0), "scan_date": session_date.isoformat(), "scan_ts": (c.scan_ts or datetime.now(timezone.utc).isoformat())}) for c in rejected_candidates[: int(limit)]],
        "prefilter_counts": pre_counts,
        "prefilter_samples": prefilter_samples,
        "thresholds_used": (dict(thresholds_used or {}) | {"monitor_first_mode": bool(monitor_first_mode)}),
        "reject_counts": pre_counts | reject_counts if isinstance(pre_counts, dict) else reject_counts,
        "data_failures": data_failures,
        "shortlisted": len(shortlisted),
    }


# -------------------- Range Reversion (V3) --------------------

def _rr_rolling_slope(y):
    import numpy as _np
    y = _np.asarray(y, dtype=float)
    if y.size < 5:
        return 0.0
    x = _np.arange(y.size, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    yy = (y - y.mean()) / (y.std() + 1e-12)
    return float(_np.polyfit(x, yy, 1)[0])

def _rr_daily_context(provider: AlpacaProvider, symbol: str, *, session_date: date, session_open: float | None) -> dict:
    # Uses ONLY real daily provider data; lagged where appropriate.
    daily = provider.get_daily_history(symbol, period="6mo")
    if daily is None or daily.empty:
        return {"prev_close": None, "avg20_vol": None}

    d = daily.copy()
    d = d.sort_index()
    d["day"] = d.index.date
    d = d[d["day"] <= session_date]
    if d.empty:
        return {"prev_close": None, "avg20_vol": None}

    close = d["Close"].astype(float)
    vol = d["Volume"].astype(float)
    opn = d["Open"].astype(float) if "Open" in d.columns else None
    high = d["High"].astype(float) if "High" in d.columns else None
    low  = d["Low"].astype(float) if "Low" in d.columns else None

    # prev close (previous day)
    prev_rows = d[d["day"] < session_date]
    prev_close = float(prev_rows["Close"].astype(float).iloc[-1]) if not prev_rows.empty else None

    # rolling lagged features (as of session_date)
    sma20 = close.rolling(20).mean().shift(1)
    sma50 = close.rolling(50).mean().shift(1)
    trend_20_50 = float((sma20.iloc[-1] > sma50.iloc[-1])) if (sma20.notna().iloc[-1] and sma50.notna().iloc[-1]) else float("nan")

    ret = close.pct_change()
    vol20 = ret.rolling(20).std().shift(1).iloc[-1]

    avg20_vol = vol.rolling(20).mean().shift(1).iloc[-1]
    avg20_dollar_vol = float(avg20_vol) * float(prev_close) if (avg20_vol is not None and prev_close) else float("nan")

    mom5 = (close.shift(1) / close.shift(6) - 1.0).iloc[-1]
    mom20 = (close.shift(1) / close.shift(21) - 1.0).iloc[-1]

    gap_pct = float("nan")
    if session_open is not None and prev_close is not None and prev_close > 0:
        gap_pct = float(session_open / prev_close - 1.0)

    atr14_pct = float("nan")
    if high is not None and low is not None:
        prev_c = close.shift(1)
        tr = pd.concat([(high-low).abs(), (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().shift(1).iloc[-1]
        if prev_close and atr14 is not None:
            atr14_pct = float(atr14 / prev_close)

    return {
        "prev_close": prev_close,
        "avg20_vol": float(avg20_vol) if pd.notna(avg20_vol) else None,
        "trend_20_50": trend_20_50,
        "vol20": float(vol20) if pd.notna(vol20) else float("nan"),
        "avg20_dollar_vol": float(avg20_dollar_vol) if pd.notna(avg20_dollar_vol) else float("nan"),
        "mom5": float(mom5) if pd.notna(mom5) else float("nan"),
        "mom20": float(mom20) if pd.notna(mom20) else float("nan"),
        "gap_pct": float(gap_pct) if pd.notna(gap_pct) else float("nan"),
        "atr14_pct": float(atr14_pct) if pd.notna(atr14_pct) else float("nan"),
    }

def build_range_reversion_plan(
    provider: AlpacaProvider,
    symbol: str,
    cfg: ORBConfig,
    *,
    session_date: date,
    range_window_min: int = 60,
    band_k: float = 2.0,
    stop_sigma_mult: float = 0.75,
    touch_lookback_min: int = 15,
) -> tuple[Candidate, dict]:
    # Fetch intraday 1m for session_date (real provider)
    if hasattr(provider, "get_bars_range"):
        intraday = provider.get_bars_range(symbol=symbol, interval="1m", from_d=session_date, to_d=session_date, include_prepost=False)
    else:
        intraday = provider.get_bars(BarsRequest(symbol=symbol, interval="1m", period="5d", include_prepost=False))
        intraday = intraday[intraday.index.date == session_date]

    if intraday is None or intraday.empty:
        raise RuntimeError("Provider returned empty intraday bars")

    df = intraday.sort_index().copy()
    # RTH filter
    # RTH filter in America/New_York (provider bars are typically UTC-indexed)
    if getattr(df.index, 'tz', None) is None:
        df = df.tz_localize('UTC')
    df = df.tz_convert(ET).between_time('09:30', '16:00')

    min_bars_required = max(45, int(range_window_min))
    if df.empty or len(df) < min_bars_required:
        raise ValueError("rr_not_enough_rth_bars")

    last_price = float(df["Close"].astype(float).iloc[-1])
    session_open = float(df["Open"].astype(float).iloc[0])

    # Rolling vwap + sigma
    tp = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3.0
    vol = df["Volume"].astype(float)
    minp = max(10, int(range_window_min // 3))

    vwap_roll = (tp * vol).rolling(range_window_min, min_periods=minp).sum() / (vol.rolling(range_window_min, min_periods=minp).sum() + 1e-12)
    sigma_roll = df["Close"].astype(float).rolling(range_window_min, min_periods=minp).std()

    lower = vwap_roll - float(band_k) * sigma_roll
    upper = vwap_roll + float(band_k) * sigma_roll

    # Pick most recent strict touch, then fall back to a near-touch if price tagged the lower band zone.
    now_ts = df.index[-1]
    touch_lookback_effective = max(int(touch_lookback_min), min(int(range_window_min), 45))
    cutoff = now_ts - timedelta(minutes=touch_lookback_effective)

    low_s = df["Low"].astype(float)
    close_s = df["Close"].astype(float)
    sigma_s = sigma_roll.astype(float).fillna(0.0)
    lower_s = lower.astype(float)

    strict_touch_mask = (low_s <= lower_s) & (df.index >= cutoff)
    touched = df.index[strict_touch_mask]
    touch_kind = "strict"

    if len(touched) == 0:
        near_touch_mask = (
            (df.index >= cutoff)
            & (sigma_s > 1e-9)
            & (low_s <= (lower_s + 0.25 * sigma_s))
            & (close_s <= (lower_s + 0.50 * sigma_s))
        )
        touched = df.index[near_touch_mask]
        touch_kind = "near"

    if len(touched) == 0:
        raise ValueError("no_recent_touch")

    t_touch = touched[-1]
    i = df.index.get_loc(t_touch)

    vi = float(vwap_roll.iloc[i])
    si = float(sigma_roll.iloc[i])
    li = float(lower.iloc[i])
    ui = float(upper.iloc[i])
    if not (si > 1e-9 and vi > 0):
        raise ValueError("rr_sigma_vwap_invalid")

    entry = li
    stop = li - float(stop_sigma_mult) * si
    risk = entry - stop
    if not (risk > 0 and cfg.min_risk_per_share <= risk <= cfg.max_risk_per_share):
        raise ValueError("risk_out_of_bounds")

    shares = int(math.floor(cfg.risk_dollars / risk)) if risk > 0 else 0
    notional = float(shares) * float(entry)
    if shares < cfg.min_shares or notional > cfg.max_notional:
        raise ValueError("position_sizing_invalid")

    # Targets:
    # - t_mid: VWAP reclaim / midline objective
    # - t_band: upper band objective
    # - t_2r: true 2R objective from entry/stop math
    t_mid = vi
    t_band = ui
    t_2r = entry + 2.0 * risk

    # Feature row (matches rr_ds_v3 base columns)
    lb = max(0, i - 30)
    win = df.iloc[lb:i+1]
    win_close = win["Close"].astype(float).to_numpy()
    win_vwap = vwap_roll.iloc[lb:i+1].astype(float).to_numpy()

    above = (win_close > win_vwap).astype(int)
    vwap_crosses = float((above[1:] != above[:-1]).sum()) if above.size >= 2 else 0.0
    slope = float(_rr_rolling_slope(win_close))
    atr_proxy = float((win["High"].astype(float) - win["Low"].astype(float)).mean())

    vol_now = float(df["Volume"].astype(float).iloc[i])
    relvol5 = vol_now / (float(win["Volume"].astype(float).tail(5).mean()) + 1e-12)
    relvol15 = vol_now / (float(win["Volume"].astype(float).tail(15).mean()) + 1e-12)

    dt_et = t_touch.tz_convert(ET) if hasattr(t_touch, "tz_convert") else t_touch
    tod_min = float((dt_et.hour * 60 + dt_et.minute) - (9 * 60 + 30))

    trades = float(df["Trades"].astype(float).iloc[i]) if "Trades" in df.columns else 0.0

    zscore = float((float(df["Close"].astype(float).iloc[i]) - vi) / (si + 1e-12))
    band_width_pct = float((ui - li) / max(1e-12, vi))
    dist_to_entry_sig = float((float(df["Close"].astype(float).iloc[i]) - entry) / (si + 1e-12))
    dist_to_stop_sig = float((float(df["Close"].astype(float).iloc[i]) - stop) / (si + 1e-12))

    # Daily context
    ctx = _rr_daily_context(provider, symbol, session_date=session_date, session_open=session_open)
    prev_close = ctx.get("prev_close")
    pct_change = ((last_price - prev_close) / prev_close * 100.0) if (prev_close and prev_close > 0) else None

    avg20_vol = ctx.get("avg20_vol")
    rvol = (float(df["Volume"].astype(float).sum()) / float(avg20_vol)) if (avg20_vol and avg20_vol > 0) else None

    today_vol = float(df["Volume"].astype(float).sum())
    today_dollar_vol = float(today_vol * last_price)

    # vwap + trend (existing helpers)
    vw = vwap(df)
    tstate = trend_state_1m(df, vw=vw, lookback=15)

    c = Candidate(
        symbol=symbol,
        data_date=str(session_date.isoformat()),
        last_price=last_price,
        pct_change=pct_change,
        rvol=rvol,
        today_dollar_vol=today_dollar_vol,
        avg20_dollar_vol=float(ctx.get("avg20_dollar_vol")) if ctx.get("avg20_dollar_vol") is not None else None,

        # Reuse OR fields to carry RR bands for UI + monitor:
        or_high=ui,
        or_low=li,
        or_range_pct=band_width_pct * 100.0,

        above_vwap=True if (last_price > float(vw.iloc[-1])) else False,
        vwap_last=float(vw.iloc[-1]),
        vwap_delta_pct=((last_price - float(vw.iloc[-1])) / float(vw.iloc[-1]) * 100.0) if float(vw.iloc[-1]) > 0 else None,
        trend_state=str(tstate.get("state")) if tstate else None,
        trend_slope_pct=float(tstate.get("slope_pct_lookback")) if tstate and tstate.get("slope_pct_lookback") is not None else None,

        best_side="long",
        entry=float(entry),
        stop=float(stop),
        target_2r=float(t_2r),
        target_3r=float(t_band),
        risk_per_share=float(risk),
        shares=int(shares),
        notional=float(notional),

        long_entry=float(entry),
        long_stop=float(stop),
        long_2r=float(t_2r),
        long_3r=float(t_band),
        long_risk_per_share=float(risk),
        long_shares=int(shares),
        long_notional=float(notional),

        short_entry=None,
        short_stop=None,
        short_2r=None,
        short_3r=None,
        short_risk_per_share=None,
        short_shares=None,
        short_notional=None,

        notes=f"RR {touch_kind} touch @ {t_touch.isoformat()} lower={li:.4f} vwap={vi:.4f} mid_target={t_mid:.4f} band_target={t_band:.4f}",
        stop_loss=float(stop),
        take_profit=float(t_2r),
        strategy="range_reversion",
        scan_ts=datetime.now(timezone.utc).isoformat(),
        touch_ts=t_touch.isoformat(),
        prior_session_touch=(t_touch.astimezone(ET).date() != session_date),
    )

    feats = {
        "zscore": zscore,
        "band_width_pct": band_width_pct,
        "vwap_crosses": vwap_crosses,
        "slope": slope,
        "atr_proxy": atr_proxy,
        "sigma": si,
        "relvol5": float(relvol5),
        "relvol15": float(relvol15),
        "tod_min": float(tod_min),
        "transactions": float(trades),
        "volume": float(vol_now),
        "trend_20_50": float(ctx.get("trend_20_50", float("nan"))),
        "vol20": float(ctx.get("vol20", float("nan"))),
        "avg20_dollar_vol": float(ctx.get("avg20_dollar_vol", float("nan"))),
        "mom5": float(ctx.get("mom5", float("nan"))),
        "mom20": float(ctx.get("mom20", float("nan"))),
        "gap_pct": float(ctx.get("gap_pct", float("nan"))),
        "atr14_pct": float(ctx.get("atr14_pct", float("nan"))),
        "dist_to_entry_sig": dist_to_entry_sig,
        "dist_to_stop_sig": dist_to_stop_sig,
        "rr_touch_kind": touch_kind,
        "rr_touch_lookback_effective": float(touch_lookback_effective),
    }
    return c, feats

def scan_range_reversion_symbols(
    symbols: List[str],
    cfg: ORBConfig,
    limit: int = 25,
    **kwargs,
) -> Dict[str, Any]:
    stream_cache = kwargs.get("stream_cache")
    streaming_only = bool(kwargs.get("streaming_only", False) or kwargs.get("runtime_streaming_only", False))
    if stream_cache is not None and streaming_only:
        return _scan_symbols_streaming_only(
            symbols,
            cfg,
            limit=limit,
            stream_cache=stream_cache,
            strategy="range_reversion",
            exec_style="reversion",
            use_ml=bool(kwargs.get("use_ml", False)),
            range_window_min=int(kwargs.get("range_window_min", 60)),
            band_k=float(kwargs.get("range_band_k", 2.0)),
            stop_sigma_mult=float(kwargs.get("rr_stop_sigma_mult", 0.75)),
            touch_lookback_min=int(kwargs.get("rr_touch_lookback_min", 15)),
        )

    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    use_ml: bool = bool(kwargs.get("use_ml", False))
    use_sentiment: bool = bool(kwargs.get("use_sentiment", False))
    sentiment_provider: str = str(kwargs.get("sentiment_provider", "auto"))
    sentiment_alpha: float = float(kwargs.get("sentiment_alpha", 0.15))
    use_catalyst: bool = bool(kwargs.get("use_catalyst", str(os.getenv("ORB_USE_CATALYST", "0")).lower() in {"1","true","yes","on"}))
    catalyst_alpha: float = float(kwargs.get("catalyst_alpha", os.getenv("ORB_CATALYST_ALPHA", 0.08)))
    catalyst_topn = kwargs.get("catalyst_topn", None)
    catalyst_lookback_hours: int = int(kwargs.get("catalyst_lookback_hours", os.getenv("ORB_CATALYST_LOOKBACK_HOURS", 72)))
    regime_profile: str = str(kwargs.get("regime_profile", os.getenv("ORB_REGIME_PROFILE", "auto"))).strip().lower() or "auto"

    range_window_min = int(kwargs.get("range_window_min", 60))
    band_k = float(kwargs.get("range_band_k", 2.0))
    stop_sigma_mult = float(kwargs.get("rr_stop_sigma_mult", 1.0))
    touch_lookback_min = int(kwargs.get("rr_touch_lookback_min", 30))

    base_provider = kwargs.get("provider") or AlpacaProvider()

    class _CachedProvider:
        def __init__(self, inner: AlpacaProvider):
            self._p = inner
            self.name = getattr(inner, "name", "alpaca")
            self._bars_cache = {}
            self._daily_cache = {}

        def get_bars(self, req: BarsRequest, timeout_s: int | None = None) -> pd.DataFrame:
            k = (req.symbol, req.interval, req.period, bool(req.include_prepost))
            if k in self._bars_cache:
                return self._bars_cache[k]
            df = self._p.get_bars(req, timeout_s=timeout_s)
            self._bars_cache[k] = df
            return df

        def get_bars_range(self, *, symbol: str, interval: str, from_d: date, to_d: date, include_prepost: bool = False, timeout_s: int | None = None) -> pd.DataFrame:
            k = ("range", symbol, interval, from_d.isoformat(), to_d.isoformat(), bool(include_prepost))
            if k in self._bars_cache:
                return self._bars_cache[k]
            df = self._p.get_bars_range(symbol=symbol, interval=interval, from_d=from_d, to_d=to_d, include_prepost=include_prepost, timeout_s=timeout_s)
            self._bars_cache[k] = df
            return df

        def get_daily_history(self, symbol: str, period: str = "6mo", timeout_s: int | None = None) -> pd.DataFrame:
            k = (symbol, period)
            if k in self._daily_cache:
                return self._daily_cache[k]
            df = self._p.get_daily_history(symbol, period=period, timeout_s=timeout_s)
            self._daily_cache[k] = df
            return df

        def __getattr__(self, item):
            return getattr(self._p, item)

    provider = _CachedProvider(base_provider)
    session_date = resolve_session_date(base_provider)

    shortlisted, pre_counts, daily_errors, prefilter_samples, thresholds_used = _prefilter_daily(base_provider, symbols, cfg)

    candidates: List[Candidate] = []
    feat_rows: List[dict] = []
    feat_by_symbol: Dict[str, dict] = {}
    data_failures = list(daily_errors)
    reject_counts = {"no_recent_touch": 0, "risk_out_of_bounds": 0, "position_sizing_invalid": 0, "intraday_error": 0}
    failure_samples_by_code: dict[str, list[dict[str, Any]]] = {}

    def _one(sym: str):
        try:
            c, feats = build_range_reversion_plan(
                provider, sym, cfg,
                session_date=session_date,
                range_window_min=range_window_min,
                band_k=band_k,
                stop_sigma_mult=stop_sigma_mult,
                touch_lookback_min=touch_lookback_min,
            )
            return sym, c, feats, None
        except ValueError as e:
            return sym, None, None, str(e)
        except Exception as e:
            return sym, None, None, f"{type(e).__name__}: {e}"

    workers = int(kwargs.get("workers", os.getenv("ORB_WORKERS", 10)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, s) for s in shortlisted]
        for fut in as_completed(futs):
            sym, c, feats, err = fut.result()
            if c is not None:
                candidates.append(c)
                feat_rows.append(feats)
                feat_by_symbol[c.symbol] = feats
            else:
                raw_err = err or "rr_plan_unknown"
                code, detail = _classify_intraday_exception(raw_err)
                if code == "no_recent_touch":
                    reject_counts["no_recent_touch"] += 1
                elif code == "risk_out_of_bounds":
                    reject_counts["risk_out_of_bounds"] += 1
                elif code == "position_sizing_invalid":
                    reject_counts["position_sizing_invalid"] += 1
                else:
                    reject_counts["intraday_error"] += 1
                fail_item = {"symbol": sym, "stage": "rr_plan", "error": detail, "code": code}
                data_failures.append(fail_item)
                _append_failure_sample(failure_samples_by_code, code=code, item=fail_item)

    # ML scoring (gold RR)
    if use_ml and candidates:
        try:
            model_path = Path(_PROJECT_ROOT) / "models" / "range_reversion_gold.pkl"
            scorer = RangeReversionGoldScorer(model_path)
            df_feats = pd.DataFrame(feat_rows)
            probs = scorer.predict_proba(df_feats)
            for c, p in zip(candidates, probs):
                c.ml_score = float(p)
                c.combined_score = float(p)
        except Exception as e:
            data_failures.append({"symbol": "__all__", "stage": "rr_ml", "error": f"{type(e).__name__}: {e}"})
            for c in candidates:
                c.ml_score = None
                c.combined_score = 0.0
    else:
        for c in candidates:
            c.ml_score = None
            c.combined_score = 0.0

    # Optional sentiment + catalyst + regime/combined score reuse (same logic as ORB)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def base_score(c: Candidate) -> float:
        # fallback if no ML
        s = 0.0
        s += min(30.0, (c.today_dollar_vol or 0.0) / 25_000_000.0 * 30.0)
        s += min(30.0, (c.rvol or 0.0) / 8.0 * 30.0)
        s += max(0.0, 20.0 - abs(c.or_range_pct - 4.0) * 4.0)
        return float(s)

    sentiment_map: Dict[str, float] = {}
    if use_sentiment and candidates:
        pre_sorted = sorted(candidates, key=lambda x: (float(x.ml_score or 0.0), base_score(x)), reverse=True) if use_ml else sorted(candidates, key=base_score, reverse=True)
        topn = int(os.getenv("ORB_SENTIMENT_TOPN", max(50, int(limit) * 3)))
        target_syms = [c.symbol for c in pre_sorted[: min(topn, len(pre_sorted))]]
        try:
            svc = SentimentService(provider=sentiment_provider)
            def _sent_one(sym: str):
                try:
                    b = svc.fetch(sym, limit=6)
                    return sym, float(b.score), None
                except Exception as e:
                    return sym, None, str(e)
            with ThreadPoolExecutor(max_workers=int(os.getenv("ORB_SENTIMENT_WORKERS", 8))) as ex:
                futs = [ex.submit(_sent_one, s) for s in target_syms]
                for fut in as_completed(futs):
                    sym, sc, err = fut.result()
                    if sc is not None:
                        sentiment_map[sym] = sc
                    else:
                        data_failures.append({"symbol": sym, "stage": "sentiment", "error": err or "unknown"})
        except Exception as e:
            data_failures.append({"symbol": "__all__", "stage": "sentiment_init", "error": str(e)})

    catalyst_map = {}
    if use_catalyst and candidates:
        pre_sorted2 = sorted(candidates, key=lambda x: (float(x.ml_score or 0.0), base_score(x)), reverse=True) if use_ml else sorted(candidates, key=base_score, reverse=True)
        ctopn = int(catalyst_topn or os.getenv("ORB_CATALYST_TOPN", max(25, int(limit) * 2)))
        target_syms = [c.symbol for c in pre_sorted2[: min(ctopn, len(pre_sorted2))]]
        try:
            csvc = CatalystService(provider=provider)
            catalyst_map = csvc.fetch_batch(target_syms, per_symbol_limit=6, lookback_hours=int(catalyst_lookback_hours))
            for sym, bundle in catalyst_map.items():
                if getattr(bundle, "error", None):
                    data_failures.append({"symbol": sym, "stage": "catalyst", "error": str(bundle.error)})
        except Exception as e:
            data_failures.append({"symbol": "__all__", "stage": "catalyst_init", "error": f"{type(e).__name__}: {e}"})
            catalyst_map = {}

    regime = _pick_regime_profile(regime_profile, candidates)
    # RR had become overfiltered in practice. Use more realistic default gates,
    # while still allowing explicit overrides from kwargs/env-driven callers.
    rr_min_ml_value = float(kwargs.get("rr_min_ml_score", 0.35)) if use_ml else 0.0
    rr_min_confidence_value = float(kwargs.get("rr_min_confidence", 45.0))
    admissible: list[Candidate] = []
    rejected_candidates: list[Candidate] = []

    for c in candidates:
        c.sentiment_score = sentiment_map.get(c.symbol)

        cb = catalyst_map.get(c.symbol) if isinstance(catalyst_map, dict) else None
        if cb is not None:
            c.catalyst_score = float(getattr(cb, "score", 0.0) or 0.0)
            c.catalyst_article_count = int(getattr(cb, "article_count", 0) or 0)
            c.catalyst_freshness_hours = getattr(cb, "freshness_hours", None)
            c.catalyst_tags = list(getattr(cb, "tags", []) or [])
        else:
            c.catalyst_score = None
            c.catalyst_article_count = None
            c.catalyst_freshness_hours = None
            c.catalyst_tags = None

        ml_base = float(c.ml_score or 0.0) if use_ml else 0.0
        rule_norm = _rr_setup_rule_score(c, feats)
        sent_term = float(sentiment_alpha) * float(c.sentiment_score or 0.0) if c.sentiment_score is not None else 0.0
        cat_term = float(catalyst_alpha) * float(c.catalyst_score or 0.0) if c.catalyst_score is not None else 0.0
        sw = regime.get("score_weights", {})
        score = (
            float(sw.get("ml", 1.0)) * ml_base +
            float(sw.get("sentiment", 1.0)) * sent_term +
            float(sw.get("catalyst", 1.0)) * cat_term +
            float(sw.get("rule", 0.0)) * rule_norm +
            float(regime.get("bias", 0.0))
        )
        c.regime_profile = str(regime.get("selected"))
        c.regime_adjustment = float(round(score - (ml_base + sent_term + cat_term), 6))
        c.combined_score = float(score)

        setup_q = rule_norm
        tape_q = _rr_tape_proxy_score(c, feats)
        align_q = _rr_alignment_score(c)
        cw = regime.get("confidence_weights", {})
        conf = (
            float(cw.get("setup", 0.35)) * setup_q +
            float(cw.get("tape", 0.20)) * tape_q +
            float(cw.get("alignment", 0.20)) * align_q +
            float(cw.get("catalyst", 0.25)) * (float(c.catalyst_score or 0.0) if c.catalyst_score is not None else 0.0)
        ) * 100.0
        c.confidence_score = float(round(conf, 2))
        c.confidence_grade = _grade(conf)

        feats = feat_by_symbol.get(c.symbol) or {}
        rr_ok, rr_reason, rr_touch_age_min, rr_chase_r = _passes_rr_actionable_long_gate(c, feats)
        c.rr_actionable_now = bool(rr_ok)
        c.rr_touch_age_min = float(rr_touch_age_min)
        c.rr_chase_r = float(rr_chase_r)

        gate_reasons: list[str] = []
        if not rr_ok and rr_reason:
            gate_reasons.append(rr_reason)
        # RR ML is currently more useful as a ranking/diagnostic signal than
        # a hard reject gate for forensic runs. Keep it visible in score_breakdown/UI,
        # but don't let the model gate hide whether actionable RR setups exist.
        # RR confidence is also diagnostic-only for now.

        c.gate_passes = not gate_reasons
        c.gate_fail_reasons = gate_reasons
        c.score_breakdown = {
            "strategy": "range_reversion",
            "ml_score": round(float(c.ml_score or 0.0), 6),
            "rule_score": round(rule_norm, 6),
            "tape_score": round(tape_q, 6),
            "alignment_score": round(align_q, 6),
            "sentiment_term": round(sent_term, 6),
            "catalyst_term": round(cat_term, 6),
            "combined_score": round(float(c.combined_score or 0.0), 6),
            "confidence_score": round(float(c.confidence_score or 0.0), 2),
            "confidence_grade": c.confidence_grade,
            "rr_touch_age_min": round(float(rr_touch_age_min), 2),
            "rr_chase_r": round(float(rr_chase_r), 4),
            "gate_passes": c.gate_passes,
        }

        if c.gate_passes:
            admissible.append(c)
        else:
            rejected_candidates.append(c)


    if not prefilter_samples:
        for sym in shortlisted[:10]:
            prefilter_samples.append({
                "symbol": sym,
                "passed": True,
                "reason": "daily_ok",
            })

    admissible.sort(key=lambda x: (float(x.combined_score or 0.0), float(x.ml_score or 0.0)), reverse=True)
    top = admissible[: int(limit)]

    for _code, _items in _build_failure_samples_by_code(data_failures, limit_per_code=5).items():
        bucket = failure_samples_by_code.setdefault(_code, [])
        for _it in _items:
            if len(bucket) < 5:
                bucket.append(_it)

    return {
        "provider": provider.name,
        "scan_date": session_date.isoformat(),
        "regime": regime,
        "debug": {
            "session_date_used": session_date.isoformat(),
            "failure_samples": data_failures[:10],
            "failure_samples_by_code": failure_samples_by_code,
        },
        "count": len(top),
        "candidates_total": len(admissible),
        "rejected_total": len(rejected_candidates),
        "candidates": [({**asdict(c), "price": float(c.last_price or 0.0), "scan_date": session_date.isoformat(), "scan_ts": (c.scan_ts or datetime.now(timezone.utc).isoformat())}) for c in top],
        "rejected_candidates": [({**asdict(c), "price": float(c.last_price or 0.0), "scan_date": session_date.isoformat(), "scan_ts": (c.scan_ts or datetime.now(timezone.utc).isoformat())}) for c in rejected_candidates[: int(limit)]],
        "prefilter_counts": pre_counts,
        "prefilter_samples": prefilter_samples,
        "thresholds_used": thresholds_used,
        "reject_counts": reject_counts,
        "data_failures": data_failures,
        "shortlisted": len(shortlisted),
    }


def _grade(x: float) -> str:
    # simple consistent grading used across scanner paths
    if x >= 85: return "A"
    if x >= 70: return "B"
    if x >= 55: return "C"
    if x >= 40: return "D"
    return "F"
