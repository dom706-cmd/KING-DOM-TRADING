
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Protocol
import math
import threading
import time
import uuid
import hashlib

from providers.base import BarsRequest
from runtime.stream_market_data import recent_bars_df, latest_trade_payload, latest_quote_payload
from sentiment.catalyst import CatalystService
from core.errors import MonitorTradeRefreshFailure, MonitorQuoteRefreshFailure, failure_string
from core.execution_plan import build_plan_state
from ml.plan_confidence import compute_plan_readiness, plan_readiness_grade as _plan_readiness_grade, extract_plan_snapshot_features, TICKER_SECTOR

ET = ZoneInfo("America/New_York")


def _now_ts() -> float:
    return time.time()


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _parse_ts(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, datetime):
        dt = v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None
    return None


def _age_ms(ts: float | None, now_ts: float | None = None) -> int | None:
    if ts is None:
        return None
    now = _now_ts() if now_ts is None else float(now_ts)
    return max(0, int(round((now - float(ts)) * 1000.0)))


def _grade(score: float) -> str:
    if score >= 92:
        return "A"
    if score >= 82:
        return "B"
    if score >= 70:
        return "C"
    return "D"


def _dedupe_keep(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _session_state(now_ts: float) -> str:
    dt = datetime.fromtimestamp(now_ts, tz=ET)
    hm = dt.hour * 60 + dt.minute
    if 4 * 60 <= hm < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= hm < 16 * 60:
        return "regular"
    if 16 * 60 <= hm < 20 * 60:
        return "afterhours"
    return "closed"


def _time_of_day_bucket(now_ts: float) -> str:
    dt = datetime.fromtimestamp(now_ts, tz=ET)
    hm = dt.hour * 60 + dt.minute
    if 9 * 60 + 30 <= hm < 10 * 60 + 15:
        return "open_impulse"
    if 10 * 60 + 15 <= hm < 11 * 60 + 30:
        return "post_open"
    if 11 * 60 + 30 <= hm < 14 * 60 + 30:
        return "midday"
    if 14 * 60 + 30 <= hm < 16 * 60:
        return "late_day"
    if hm < 9 * 60 + 30:
        return "premarket"
    return "afterhours"


def _best_seed_score(c: dict[str, Any]) -> float:
    for key in ("combined_score", "score", "ml_score", "confidence_score"):
        try:
            v = c.get(key)
            if v is not None:
                return float(v)
        except Exception:
            continue
    return 0.0


def _compute_monitor_live_score(st: "MonitorSymbolState", *, playbook: "BasePlaybook", live_state: dict[str, Any], context: dict[str, Any], context_score: float, flags: list[str]) -> float:
    freshness_score = 10.0 if st.tape_live else -20.0
    catalyst_component = float(st.catalyst_score or 0.0) * 18.0
    decay_penalty = 0.0
    if st.catalyst_freshness_hours is not None and st.catalyst_freshness_hours > 6.0:
        decay_penalty += min(15.0, (float(st.catalyst_freshness_hours) - 6.0) * 2.0)
    risk_penalty = 0.0
    if 'wide_spread' in flags or 'context_misaligned' in flags:
        risk_penalty += 10.0
    return playbook.score(st, live_state, context) + freshness_score + catalyst_component - decay_penalty - risk_penalty


def _monitor_live_confidence_reasons(st: "MonitorSymbolState", reasons: list[str]) -> list[str]:
    reasons_out = list(reasons)
    if st.tape_live:
        reasons_out.append('tape_live')
    if st.above_vwap_live is True and st.best_side == 'long':
        reasons_out.append('vwap_aligned_long')
    if st.above_vwap_live is False and st.best_side == 'short':
        reasons_out.append('vwap_aligned_short')
    if st.catalyst_freshness_hours is not None and st.catalyst_freshness_hours <= 1.5:
        reasons_out.append('news_fresh')
    return _dedupe_keep(reasons_out)


def _near_trigger_live(st: "MonitorSymbolState") -> bool:
    if not st.tape_live:
        return False
    if st.entry is None or st.risk_per_share in (None, 0.0) or st.price is None:
        return False
    if st.long_triggered_live or st.short_triggered_live:
        return False
    try:
        return abs(float(st.retest_distance_r or 999.0)) <= 0.20
    except Exception:
        return False


def _alert_bucket_for_state(st: "MonitorSymbolState", state: str) -> str:
    state_norm = str(state or "watch").strip().lower()
    if state_norm == "triggered":
        return "triggered"
    if state_norm in {"arming", "confirmed", "touch_wait_confirm"}:
        return "ready"
    if state_norm in {"failed", "extended"}:
        return "suppressed"
    if _near_trigger_live(st):
        return "near_trigger"
    return "watch"




def _trade_refresh_update(sym: str, *, provider: Any, stream_cache: Any, snapshot: dict | None = None) -> dict[str, Any]:
    stream_err = None
    try:
        payload = latest_trade_payload(stream_cache, sym, max_age_sec=30.0)
        return {
            "price": _safe_float(payload.get("price")),
            "trade_ts": _parse_ts(payload.get("timestamp")),
            "trade_size": _safe_int(payload.get("size")),
        }
    except Exception as e:
        stream_err = e

    # Use pre-fetched batch snapshot if available — avoids a per-symbol REST call
    if snapshot is not None:
        lt = snapshot.get("latest_trade") or snapshot.get("trade") or {}
        price = _safe_float(lt.get("price") or lt.get("p"))
        if price is not None:
            return {
                "price": price,
                "trade_ts": _parse_ts(lt.get("timestamp") or lt.get("t")),
                "trade_size": _safe_int(lt.get("size") or lt.get("s")),
            }

    provider_err = None
    try:
        payload = provider.get_latest_trade(sym)
        return {
            "price": _safe_float(payload.get("price")),
            "trade_ts": _parse_ts(payload.get("timestamp")),
            "trade_size": None,
        }
    except Exception as e:
        provider_err = e

    msg = f"stream={type(stream_err).__name__ if stream_err else 'None'}:{stream_err} | provider={type(provider_err).__name__ if provider_err else 'None'}:{provider_err}"
    raise MonitorTradeRefreshFailure(
        code="trade_refresh_failed",
        message=msg,
        stage="monitor_trade_refresh",
        symbol=sym,
        cause_type=type(provider_err).__name__ if provider_err is not None else (type(stream_err).__name__ if stream_err is not None else "RuntimeError"),
    ) from (provider_err or stream_err)


def _quote_refresh_update(sym: str, *, provider: Any, stream_cache: Any, snapshot: dict | None = None) -> dict[str, Any]:
    stream_err = None
    try:
        payload = latest_quote_payload(stream_cache, sym, max_age_sec=30.0)
        return {
            "bid": _safe_float(payload.get("bid")),
            "ask": _safe_float(payload.get("ask")),
            "bid_size": _safe_int(payload.get("bid_size")),
            "ask_size": _safe_int(payload.get("ask_size")),
            "quote_ts": _parse_ts(payload.get("timestamp") if payload.get("timestamp") is not None else payload.get("quote_ts")),
        }
    except Exception as e:
        stream_err = e

    # Use pre-fetched batch snapshot if available — avoids a per-symbol REST call
    if snapshot is not None:
        lq = snapshot.get("latest_quote") or snapshot.get("quote") or {}
        bid = _safe_float(lq.get("bid_price") or lq.get("bp"))
        ask = _safe_float(lq.get("ask_price") or lq.get("ap"))
        if bid is not None or ask is not None:
            return {
                "bid": bid,
                "ask": ask,
                "bid_size": _safe_int(lq.get("bid_size") or lq.get("bs")),
                "ask_size": _safe_int(lq.get("ask_size") or lq.get("as")),
                "quote_ts": _parse_ts(lq.get("timestamp") or lq.get("t")),
            }

    provider_err = None
    try:
        payload = provider.get_latest_quote(sym)
        return {
            "bid": _safe_float(payload.get("bid_price", payload.get("bid"))),
            "ask": _safe_float(payload.get("ask_price", payload.get("ask"))),
            "bid_size": _safe_int(payload.get("bid_size")),
            "ask_size": _safe_int(payload.get("ask_size")),
            "quote_ts": _parse_ts(payload.get("timestamp") if payload.get("timestamp") is not None else payload.get("quote_ts")),
        }
    except Exception as e:
        provider_err = e

    msg = f"stream={type(stream_err).__name__ if stream_err else 'None'}:{stream_err} | provider={type(provider_err).__name__ if provider_err else 'None'}:{provider_err}"
    raise MonitorQuoteRefreshFailure(
        code="quote_refresh_failed",
        message=msg,
        stage="monitor_quote_refresh",
        symbol=sym,
        cause_type=type(provider_err).__name__ if provider_err is not None else (type(stream_err).__name__ if stream_err is not None else "RuntimeError"),
    ) from (provider_err or stream_err)


@dataclass
class MonitorAlertEvent:
    event_id: str
    monitor_id: str
    symbol: str
    playbook: str
    from_state: str
    to_state: str
    event_type: str
    event_ts: float
    price: float | None
    bid: float | None
    ask: float | None
    spread_pct: float | None
    vwap_delta_pct: float | None
    live_chase_r: float | None
    catalyst_score: float | None
    context_score: float | None
    dedupe_key: str
    alert_bucket: str = "watch"
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_api(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "monitor_id": self.monitor_id,
            "symbol": self.symbol,
            "playbook": self.playbook,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "event_type": self.event_type,
            "event_ts": self.event_ts,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": self.spread_pct,
            "vwap_delta_pct": self.vwap_delta_pct,
            "live_chase_r": self.live_chase_r,
            "catalyst_score": self.catalyst_score,
            "context_score": self.context_score,
            "dedupe_key": self.dedupe_key,
            "alert_bucket": self.alert_bucket,
            "reasons": list(self.reasons),
            "flags": list(self.flags),
        }


@dataclass
class MonitorSymbolState:
    symbol: str
    playbook: str = "open_drive_orb"
    seed_source: str = "scan"

    # scan/meta
    scan_rank: int | None = None
    scan_candidate: dict[str, Any] = field(default_factory=dict)
    strategy: str | None = None
    exec_style: str | None = None
    best_side: str | None = None

    # plan
    entry: float | None = None
    stop_loss: float | None = None
    target_2r: float | None = None
    target_3r: float | None = None
    risk_per_share: float | None = None
    or_high: float | None = None
    or_low: float | None = None
    vwap_last: float | None = None
    scan_price: float | None = None

    # scan scores
    scan_seed_score: float | None = None
    combined_score: float | None = None
    ml_score: float | None = None
    model_bucket: str | None = None
    confidence_score: float | None = None
    confidence_grade: str | None = None
    regime_profile: str | None = None
    p_2r_30m: float | None = None
    probability_source: str | None = None
    gate_passes: bool | None = None
    gate_fail_reasons: list[str] = field(default_factory=list)
    monitor_seed: bool | None = None
    monitor_seed_reasons: list[str] = field(default_factory=list)
    orb_retest_ready: bool | None = None
    tradable_now: bool | None = None
    trade_ready_passes: bool | None = None

    # catalyst / news
    catalyst_score: float | None = None
    catalyst_confidence: float | None = None
    catalyst_article_count: int | None = None
    catalyst_freshness_hours: float | None = None
    catalyst_tags: list[str] = field(default_factory=list)
    catalyst_headlines: list[str] = field(default_factory=list)
    latest_news_at: float | None = None
    news_source: str | None = None

    # live tape
    price: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    trade_ts: float | None = None
    quote_ts: float | None = None
    spread_pct: float | None = None
    quote_age_ms: int | None = None
    trade_age_ms: int | None = None
    above_vwap_live: bool | None = None
    vwap_delta_pct_live: float | None = None
    live_chase_r: float | None = None
    retest_distance_r: float | None = None

    # live state
    long_triggered_live: bool | None = None
    short_triggered_live: bool | None = None
    retest_confirmed_live: bool = False
    tape_live: bool = False
    tape_live_reason: str | None = None
    tape_deterioration: str | None = None
    market_session_state: str = "unknown"
    time_of_day_bucket: str = "unknown"
    near_trigger_live: bool = False

    # state machine
    previous_state: str = "watch"
    monitor_state: str = "watch"
    alert_state_bucket: str = "watch"
    last_transition_ts: float | None = None
    cooldown_until_ts: float | None = None
    alert_fired_count: int = 0
    just_transitioned: bool = False
    terminal_lock: bool = False
    terminal_reason: str | None = None
    terminal_state_ts: float | None = None

    # scoring / diagnostics
    context_score: float = 0.0
    live_score: float = 0.0
    live_confidence_score: float = 0.0
    live_confidence_grade: str = "D"
    live_confidence_reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    decision: str = "WAIT"
    decision_notes: list[str] = field(default_factory=list)
    decision_hint: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    promoted_by_news: bool = False
    promoted_by_monitor_transition: bool = False

    # touch history
    last_entry_touch_ts: float | None = None
    last_trigger_ts: float | None = None
    last_refresh_at: float | None = None
    last_error: str | None = None

    # live ML scoring (watchlist_plan only)
    ml_score_attempted: bool = False   # True once we've tried scoring — prevents re-queuing every tick

    # Plan confidence (watchlist_plan playbook only)
    plan_readiness_score: float | None = None
    plan_readiness_grade_str: str = "D"
    plan_readiness_breakdown: dict[str, Any] = field(default_factory=dict)
    last_snapshot_ts: float | None = None   # throttle snapshot logging
    price_history: list[tuple[float, float]] = field(default_factory=list)  # [(price, ts), ...] ring buffer
    session_archetype: str = "mixed"        # from context engine — shapes Plan Readiness weights
    same_sector_count: int = 0              # other watchlist_plan symbols in same sector (concentration penalty)
    aggressor_history: list[tuple[float, float]] = field(default_factory=list)  # [(+1 buy/-1 sell, ts)] 60s rolling
    bid_ask_imbalance: float | None = None  # current L2 snapshot: (bid_size - ask_size) / (bid_size + ask_size)
    order_flow_score: float | None = None   # combined L2 + aggressor score (0-1)
    trade_size: int | None = None           # shares on last trade print (from Alpaca stream)
    volume_at_stop: float = 0.0            # cumulative shares printed at/through stop since first hit
    stop_hit_ticks: int = 0               # consecutive ticks at/through stop (reset when price moves away)

    @staticmethod
    def _round_price(v: Any) -> Any:
        if v is None:
            return None
        try:
            return round(float(v), 4)
        except Exception:
            return v

    def to_api(self) -> dict[str, Any]:
        trigger_side = None
        if self.long_triggered_live:
            trigger_side = "long"
        elif self.short_triggered_live:
            trigger_side = "short"
        _rp = self._round_price
        return {
            "symbol": self.symbol,
            "playbook": self.playbook,
            "seed_source": self.seed_source,
            "scan_rank": self.scan_rank,
            "strategy": self.strategy,
            "exec_style": self.exec_style,
            "best_side": self.best_side,
            "entry": _rp(self.entry),
            "stop_loss": _rp(self.stop_loss),
            "target_2r": _rp(self.target_2r),
            "target_3r": _rp(self.target_3r),
            "risk_per_share": _rp(self.risk_per_share),
            "or_high": _rp(self.or_high),
            "or_low": _rp(self.or_low),
            "vwap_last": _rp(self.vwap_last),
            "scan_price": _rp(self.scan_price),
            "combined_score": self.combined_score,
            "ml_score": self.ml_score,
            "model_bucket": self.model_bucket,
            "confidence_score": self.confidence_score,
            "confidence_grade": self.confidence_grade,
            "regime_profile": self.regime_profile,
            "p_2r_30m": self.p_2r_30m,
            "probability_source": self.probability_source,
            "gate_passes": self.gate_passes,
            "gate_fail_reasons": list(self.gate_fail_reasons),
            "monitor_seed": self.monitor_seed,
            "monitor_seed_reasons": list(self.monitor_seed_reasons),
            "orb_retest_ready": self.orb_retest_ready,
            "tradable_now": self.tradable_now,
            "trade_ready_passes": self.trade_ready_passes,
            "catalyst_score": self.catalyst_score,
            "catalyst_confidence": self.catalyst_confidence,
            "catalyst_article_count": self.catalyst_article_count,
            "catalyst_freshness_hours": self.catalyst_freshness_hours,
            "catalyst_tags": list(self.catalyst_tags),
            "catalyst_headlines": list(self.catalyst_headlines),
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "spread_pct": self.spread_pct,
            "quote_age_ms": self.quote_age_ms,
            "trade_age_ms": self.trade_age_ms,
            "above_vwap": self.above_vwap_live,
            "vwap_delta_pct": self.vwap_delta_pct_live,
            "live_chase_r": self.live_chase_r,
            "retest_distance_r": self.retest_distance_r,
            "long_triggered": self.long_triggered_live,
            "short_triggered": self.short_triggered_live,
            "retest_confirmed": self.retest_confirmed_live,
            "trigger_side": trigger_side,
            "monitor_state": self.monitor_state,
            "previous_state": self.previous_state,
            "last_transition_ts": self.last_transition_ts,
            "cooldown_until_ts": self.cooldown_until_ts,
            "market_session_state": self.market_session_state,
            "time_of_day_bucket": self.time_of_day_bucket,
            "tape_live": self.tape_live,
            "tape_live_reason": self.tape_live_reason,
            "tape_deterioration": self.tape_deterioration,
            "near_trigger": self.near_trigger_live,
            "alert_state_bucket": self.alert_state_bucket,
            "context_score": self.context_score,
            "live_score": self.live_score,
            "live_confidence_score": self.live_confidence_score,
            "live_confidence_grade": self.live_confidence_grade,
            "live_confidence_reasons": list(self.live_confidence_reasons),
            "risk_flags": list(self.risk_flags),
            "diagnostics": dict(self.diagnostics),
            "plan_state": self.decision,
            "decision": self.decision,
            "decision_notes": list(self.decision_notes),
            "decision_hint": self.decision_hint,
            "rejection_reasons": list(self.rejection_reasons),
            "execution_plan": {
                "side": self.best_side,
                "entry": _rp(self.entry),
                "current_price": _rp(self.price),
                "stop_loss": _rp(self.stop_loss),
                "target_2r": _rp(self.target_2r),
                "target_3r": _rp(self.target_3r),
                "risk_per_share": _rp(self.risk_per_share),
                "p_2r_30m": self.p_2r_30m,
                "probability_source": self.probability_source,
                "tradable_now": self.tradable_now,
                "orb_retest_ready": self.orb_retest_ready,
                "gate_passes": self.gate_passes,
                "gate_fail_reasons": list(self.gate_fail_reasons),
                "monitor_seed_reasons": list(self.monitor_seed_reasons),
            },
            "tradeability": {
                "tape_live": self.tape_live,
                "tape_live_reason": self.tape_live_reason,
                "spread_pct": self.spread_pct,
                "quote_age_ms": self.quote_age_ms,
                "trade_age_ms": self.trade_age_ms,
                "near_trigger": self.near_trigger_live,
                "trigger_side": trigger_side,
                "market_session_state": self.market_session_state,
                "time_of_day_bucket": self.time_of_day_bucket,
            },
            "promoted_by_news": self.promoted_by_news,
            "promoted_by_monitor_transition": self.promoted_by_monitor_transition,
            "last_refresh_at": self.last_refresh_at,
            "last_error": self.last_error,
            "updated_at": self.last_refresh_at or _now_ts(),
            "plan_readiness_score": self.plan_readiness_score,
            "plan_readiness_grade": self.plan_readiness_grade_str,
            "plan_readiness_breakdown": dict(self.plan_readiness_breakdown),
            "bid_ask_imbalance": round(self.bid_ask_imbalance, 3) if self.bid_ask_imbalance is not None else None,
            "order_flow_score": round(self.order_flow_score, 3) if self.order_flow_score is not None else None,
        }


@dataclass
class MonitorSession:
    monitor_id: str
    job_id: str | None
    feed_requested: str
    feed_used: str
    source: str
    mode: str = "scanner"
    started_at: float = field(default_factory=_now_ts)
    updated_at: float = field(default_factory=_now_ts)
    running: bool = True
    refresh_count: int = 0
    last_replay_persist_ts: float = 0.0
    min_refresh_interval_s: float = 1.0
    symbols: dict[str, MonitorSymbolState] = field(default_factory=dict)
    alerts: list[MonitorAlertEvent] = field(default_factory=list)
    diagnostics: dict[str, int] = field(default_factory=lambda: {
        "rejected_stale_timing": 0,
        "rejected_spread": 0,
        "rejected_no_live_confirmation": 0,
        "rejected_no_catalyst_freshness": 0,
        "rejected_multi_timeframe_disagreement": 0,
        "promoted_by_news": 0,
        "promoted_by_monitor_transition": 0,
        "demoted_by_decay": 0,
    })
    failure_samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    stream_enabled: bool = False
    stream_connected: bool | None = None
    stream_error: str | None = None
    long_only: bool = False
    short_only: bool = False
    watch_mode: str = "scanner"
    seed_symbols: list[str] = field(default_factory=list)
    promotion_candidates: list[str] = field(default_factory=list)
    last_news_refresh_ts: float | None = None
    latest_context: dict[str, Any] = field(default_factory=dict)
    seed_errors: dict[str, str] = field(default_factory=dict)

    def add_diagnostic(self, key: str, item: dict[str, Any] | None = None) -> None:
        self.diagnostics[key] = int(self.diagnostics.get(key, 0)) + 1
        if item is not None:
            bucket = self.failure_samples.setdefault(key, [])
            if len(bucket) < 10:
                bucket.append(item)


class BasePlaybook:
    name = "base"

    def compute_live_state(self, st: MonitorSymbolState, context: dict[str, Any], now_ts: float) -> dict[str, Any]:
        return {}

    def transition(self, st: MonitorSymbolState, live_state: dict[str, Any], now_ts: float) -> tuple[str, list[str], list[str]]:
        return st.monitor_state, [], []

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        return 0.0


class OpenDriveORBPlaybook(BasePlaybook):
    name = "open_drive_orb"

    def compute_live_state(self, st: MonitorSymbolState, context: dict[str, Any], now_ts: float) -> dict[str, Any]:
        return {
            "fresh": (st.tape_live and (st.trade_age_ms or 999999) <= 1500 and (st.quote_age_ms or 999999) <= 1500),
            "near_entry": (abs(float(st.retest_distance_r or 999.0)) <= 0.15),
        }

    def transition(self, st: MonitorSymbolState, live_state: dict[str, Any], now_ts: float) -> tuple[str, list[str], list[str]]:
        reasons: list[str] = []
        flags: list[str] = []
        state = "watch"
        if st.cooldown_until_ts and now_ts < st.cooldown_until_ts:
            state = "cooldown"
            flags.append("cooldown_active")
            return state, reasons, flags
        if not st.tape_live:
            state = "watch"
            flags.append(st.tape_live_reason or "tape_not_live")
            return state, reasons, flags
        if st.market_session_state != "regular":
            state = "watch"
            flags.append("session_inactive")
            return state, reasons, flags
        if st.best_side == "long":
            ctx = st.scan_candidate or {}
            dist_from_prev_close_pct = _safe_float(ctx.get("dist_from_prev_close_pct"))
            prev_day_change_pct = _safe_float(ctx.get("prev_day_change_pct"))
            dist_above_prev_day_high_pct = _safe_float(ctx.get("dist_above_prev_day_high_pct"))
            dist_above_20d_high_pct = _safe_float(ctx.get("dist_above_20d_high_pct"))

            if st.price is not None and st.stop_loss is not None and float(st.price) <= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]
            if st.price is not None and st.or_low is not None and float(st.price) < float(st.or_low):
                return "failed", reasons, ["trigger_failed"]
            if st.live_chase_r is not None and st.live_chase_r > 0.55:
                return "extended", reasons, ["extended"]

            if dist_from_prev_close_pct is not None and dist_from_prev_close_pct >= 18.0:
                return "extended", reasons, ["daily_extension"]
            if (
                prev_day_change_pct is not None and prev_day_change_pct >= 12.0 and
                dist_from_prev_close_pct is not None and dist_from_prev_close_pct >= 6.0
            ):
                return "extended", reasons, ["second_day_extension"]
            if (
                dist_above_prev_day_high_pct is not None and dist_above_prev_day_high_pct >= 4.0 and
                dist_from_prev_close_pct is not None and dist_from_prev_close_pct >= 8.0
            ):
                return "extended", reasons, ["above_prev_day_high_extended"]
            if (
                dist_above_20d_high_pct is not None and dist_above_20d_high_pct >= 5.0 and
                dist_from_prev_close_pct is not None and dist_from_prev_close_pct >= 8.0
            ):
                return "extended", reasons, ["above_20d_high_extended"]

            if st.long_triggered_live and live_state.get("fresh"):
                reasons.append("trigger_intact_long")
                return "triggered", reasons, flags
            if live_state.get("near_entry") and st.above_vwap_live:
                reasons.append("arming_long")
                return "arming", reasons, flags
            return "watch", reasons, flags
        if st.best_side == "short":
            if st.price is not None and st.stop_loss is not None and float(st.price) >= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]
            if st.price is not None and st.or_high is not None and float(st.price) > float(st.or_high):
                return "failed", reasons, ["trigger_failed"]
            if st.live_chase_r is not None and st.live_chase_r < -0.55:
                return "extended", reasons, ["extended"]
            if st.short_triggered_live and live_state.get("fresh"):
                reasons.append("trigger_intact_short")
                return "triggered", reasons, flags
            if live_state.get("near_entry") and st.above_vwap_live is False:
                reasons.append("arming_short")
                return "arming", reasons, flags
            return "watch", reasons, flags
        return state, reasons, flags

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        score = float(st.scan_seed_score or 0.0)
        if st.monitor_state == "triggered":
            score += 35.0
        elif st.monitor_state == "arming":
            score += 20.0
        elif st.monitor_state == "extended":
            score -= 20.0
        elif st.monitor_state == "failed":
            score -= 35.0
        if st.tape_live:
            score += 8.0
        if st.spread_pct is not None:
            if st.spread_pct <= 0.20:
                score += 8.0
            elif st.spread_pct <= 0.50:
                score += 2.0
            else:
                score -= 12.0
        if st.best_side == "long" and st.above_vwap_live:
            score += 6.0
        if st.best_side == "short" and st.above_vwap_live is False:
            score += 6.0
        score += float(context.get("risk_on_score") or 0.0) * (4.0 if st.best_side == "long" else -4.0)
        return score


class RetestReclaimPlaybook(BasePlaybook):
    name = "retest_reclaim_orb"

    def compute_live_state(self, st: MonitorSymbolState, context: dict[str, Any], now_ts: float) -> dict[str, Any]:
        near_entry = abs(float(st.retest_distance_r or 999.0)) <= 0.12
        return {"near_entry": near_entry, "fresh": st.tape_live}

    def transition(self, st: MonitorSymbolState, live_state: dict[str, Any], now_ts: float) -> tuple[str, list[str], list[str]]:
        reasons: list[str] = []
        flags: list[str] = []
        if st.cooldown_until_ts and now_ts < st.cooldown_until_ts:
            return "cooldown", reasons, ["cooldown_active"]
        if not st.tape_live:
            return "watch", reasons, [st.tape_live_reason or "tape_not_live"]
        if live_state.get("near_entry"):
            st.last_entry_touch_ts = now_ts
            if not st.retest_confirmed_live:
                return "touch_wait_confirm", reasons, ["touch_recorded"]
        if st.best_side == "long":
            if st.last_entry_touch_ts and st.long_triggered_live and st.above_vwap_live:
                reasons.append("reclaim_confirmed")
                return "confirmed", reasons, flags
            if st.monitor_state == "confirmed" and st.long_triggered_live:
                reasons.append("triggered_after_retest")
                return "triggered", reasons, flags
            if st.price is not None and st.stop_loss is not None and float(st.price) <= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]
        elif st.best_side == "short":
            if st.last_entry_touch_ts and st.short_triggered_live and st.above_vwap_live is False:
                reasons.append("lost_vwap_confirmed")
                return "confirmed", reasons, flags
            if st.monitor_state == "confirmed" and st.short_triggered_live:
                reasons.append("triggered_after_retest")
                return "triggered", reasons, flags
            if st.price is not None and st.stop_loss is not None and float(st.price) >= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]
        return "watch", reasons, flags

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        score = float(st.scan_seed_score or 0.0)
        if st.monitor_state == "confirmed":
            score += 26.0
        elif st.monitor_state == "triggered":
            score += 34.0
        elif st.monitor_state == "touch_wait_confirm":
            score += 12.0
        score += float(st.context_score or 0.0) * 6.0
        return score


class MiddayContinuationPlaybook(OpenDriveORBPlaybook):
    name = "midday_continuation"

    def transition(self, st: MonitorSymbolState, live_state: dict[str, Any], now_ts: float) -> tuple[str, list[str], list[str]]:
        state, reasons, flags = super().transition(st, live_state, now_ts)
        if st.time_of_day_bucket not in {"midday", "late_day"} and state in {"arming", "triggered"}:
            return "watch", reasons, flags + ["midday_window_only"]
        return state, reasons, flags

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        score = super().score(st, live_state, context)
        if st.time_of_day_bucket == "midday":
            score += 8.0
        return score


class RangeReversionPlaybook(BasePlaybook):
    name = "range_reversion"

    def compute_live_state(self, st: MonitorSymbolState, context: dict[str, Any], now_ts: float) -> dict[str, Any]:
        return {"near_entry": abs(float(st.retest_distance_r or 999.0)) <= 0.15}

    def transition(self, st: MonitorSymbolState, live_state: dict[str, Any], now_ts: float) -> tuple[str, list[str], list[str]]:
        reasons: list[str] = []
        flags: list[str] = []
        if not st.tape_live:
            return "wait_dip", reasons, [st.tape_live_reason or "tape_not_live"]
        if live_state.get("near_entry"):
            st.last_entry_touch_ts = now_ts
            return "touch_wait_confirm", reasons, ["touch_recorded"]
        if st.best_side == "long":
            if st.last_entry_touch_ts and st.price is not None and st.entry is not None and float(st.price) >= float(st.entry):
                return "confirmed", ["reversion_confirmed"], flags
            if st.price is not None and st.stop_loss is not None and float(st.price) <= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]
        elif st.best_side == "short":
            if st.last_entry_touch_ts and st.price is not None and st.entry is not None and float(st.price) <= float(st.entry):
                return "confirmed", ["reversion_confirmed"], flags
            if st.price is not None and st.stop_loss is not None and float(st.price) >= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]
        return "wait_dip", reasons, flags

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        score = float(st.scan_seed_score or 0.0)
        if st.monitor_state == "confirmed":
            score += 28.0
        if st.monitor_state == "failed":
            score -= 25.0
        return score


class CatalystNewsIgnitionPlaybook(OpenDriveORBPlaybook):
    name = "catalyst_news_ignition"

    def transition(self, st: MonitorSymbolState, live_state: dict[str, Any], now_ts: float) -> tuple[str, list[str], list[str]]:
        if st.catalyst_freshness_hours is None:
            return "watch", [], ["no_catalyst_freshness"]
        if st.catalyst_freshness_hours > 4.0:
            return "stale_news", [], ["news_stale"]
        state, reasons, flags = super().transition(st, live_state, now_ts)
        if state in {"arming", "triggered"}:
            reasons.append("news_active")
        return state, reasons, flags

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        score = super().score(st, live_state, context)
        score += float(st.catalyst_score or 0.0) * 25.0
        if st.catalyst_freshness_hours is not None:
            score += max(0.0, 12.0 - float(st.catalyst_freshness_hours) * 3.0)
        return score


class SympathyContinuationPlaybook(OpenDriveORBPlaybook):
    name = "sympathy_continuation"

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        score = super().score(st, live_state, context)
        score += float(context.get("cohort_tape_quality") or 0.0) * 15.0
        return score


class WatchlistPlanPlaybook(BasePlaybook):
    """
    Playbook for trader-defined watchlist plans.
    Uses pre-set entry/stop/target levels instead of ORB-derived ones.
    State machine is simplified: watch → arming → triggered → failed.
    Scoring is driven by plan_confidence.compute_plan_readiness() rather
    than ORB expansion/trigger logic.
    """
    name = "watchlist_plan"

    def compute_live_state(self, st: MonitorSymbolState, context: dict[str, Any], now_ts: float) -> dict[str, Any]:
        entry = st.entry
        stop  = st.stop_loss
        price = st.price
        risk  = abs(float(entry or 0) - float(stop or 0)) if entry is not None and stop is not None else None

        entry_distance_r = None
        if entry is not None and risk and risk > 1e-6 and price is not None:
            if st.best_side == "long":
                entry_distance_r = (entry - price) / risk
            else:
                entry_distance_r = (price - entry) / risk

        near_entry = (entry_distance_r is not None and 0.0 < entry_distance_r <= 0.50)
        at_entry   = (entry_distance_r is not None and entry_distance_r <= 0.0)

        return {
            "entry_distance_r": entry_distance_r,
            "near_entry":       near_entry,
            "at_entry":         at_entry,
            "fresh":            (st.tape_live and (st.trade_age_ms or 999999) <= 1500 and (st.quote_age_ms or 999999) <= 1500),
        }

    def transition(self, st: MonitorSymbolState, live_state: dict[str, Any], now_ts: float) -> tuple[str, list[str], list[str]]:
        reasons: list[str] = []
        flags:   list[str] = []

        if st.cooldown_until_ts and now_ts < st.cooldown_until_ts:
            return "cooldown", reasons, ["cooldown_active"]

        # Stop level hit
        if st.price is not None and st.stop_loss is not None:
            if st.best_side == "long"  and float(st.price) <= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]
            if st.best_side == "short" and float(st.price) >= float(st.stop_loss):
                return "failed", reasons, ["stopped_out"]

        # Entry level reached — trader's pre-planned entry is now live
        if live_state.get("at_entry"):
            reasons.append("entry_reached")
            return "triggered", reasons, flags

        # Approaching entry — within 0.5R
        if live_state.get("near_entry"):
            reasons.append("approaching_entry")
            return "arming", reasons, flags

        return "watch", reasons, flags

    def score(self, st: MonitorSymbolState, live_state: dict[str, Any], context: dict[str, Any]) -> float:
        # Scoring handled separately via compute_plan_readiness; base contribution is 0
        return 0.0


PLAYBOOKS: dict[str, BasePlaybook] = {
    OpenDriveORBPlaybook.name: OpenDriveORBPlaybook(),
    RetestReclaimPlaybook.name: RetestReclaimPlaybook(),
    MiddayContinuationPlaybook.name: MiddayContinuationPlaybook(),
    RangeReversionPlaybook.name: RangeReversionPlaybook(),
    CatalystNewsIgnitionPlaybook.name: CatalystNewsIgnitionPlaybook(),
    SympathyContinuationPlaybook.name: SympathyContinuationPlaybook(),
    WatchlistPlanPlaybook.name: WatchlistPlanPlaybook(),
}


class LiveMonitorManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, MonitorSession] = {}
        self._provider = None
        self._stream_cache = None
        self._store = None
        self._context_engine = None
        self._pubsub = None
        self._catalyst: CatalystService | None = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._refresh_interval_s = max(0.5, float(__import__("os").getenv("ORB_MONITOR_REFRESH_S", "1.0")))
        self._news_interval_s = max(10.0, float(__import__("os").getenv("ORB_NEWS_REFRESH_S", "20.0")))
        self._cooldowns = {
            "open_drive_orb": 90.0,
            "retest_reclaim_orb": 120.0,
            "midday_continuation": 180.0,
            "range_reversion": 180.0,
            "catalyst_news_ignition": 300.0,
            "sympathy_continuation": 180.0,
        }

    def configure_runtime(self, *, provider: Any, stream_cache: Any = None, store: Any = None, context_engine: Any = None, pubsub: Any = None) -> None:
        self._provider = provider
        self._stream_cache = stream_cache
        self._store = store
        self._context_engine = context_engine
        self._pubsub = pubsub
        self._catalyst = CatalystService(provider) if provider is not None else None
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(target=self._run_loop, name="live-monitor", daemon=True)
            self._worker.start()

    def stop_all(self) -> None:
        self._stop.set()
        th = self._worker
        if th and th.is_alive():
            th.join(timeout=5.0)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            mids: list[str] = []
            with self._lock:
                mids = [mid for mid, sess in self._sessions.items() if sess.running]
            for mid in mids:
                try:
                    self.refresh(monitor_id=mid, provider=self._provider, stream_cache=self._stream_cache, force=False)
                except Exception:
                    continue
            self._stop.wait(self._refresh_interval_s)

    def _make_seed_state(self, c: dict[str, Any], *, idx: int, playbook: str, seed_source: str = "scan") -> MonitorSymbolState:
        side = str(c.get("best_side") or "").strip().lower() or None
        entry = _safe_float(c.get("entry"))
        if entry is None:
            entry = _safe_float(c.get("long_entry") if side == "long" else c.get("short_entry"))
        stop = _safe_float(c.get("stop_loss"))
        if stop is None:
            stop = _safe_float(c.get("stop") if c.get("stop") is not None else (c.get("long_stop") if side == "long" else c.get("short_stop")))
        target_2r = _safe_float(c.get("target_2r"))
        if target_2r is None:
            target_2r = _safe_float(c.get("take_profit") if c.get("take_profit") is not None else (c.get("long_2r") if side == "long" else c.get("short_2r")))
        target_3r = _safe_float(c.get("target_3r"))
        if target_3r is None:
            target_3r = _safe_float(c.get("long_3r") if side == "long" else c.get("short_3r"))
        seed = _best_seed_score(c)
        return MonitorSymbolState(
            symbol=str(c.get("symbol") or "").strip().upper(),
            playbook=playbook,
            seed_source=seed_source,
            scan_rank=idx,
            scan_candidate=dict(c),
            strategy=str(c.get("strategy") or "orb"),
            exec_style=str(c.get("exec_style") or c.get("execution") or "retest"),
            best_side=side,
            entry=entry,
            stop_loss=stop,
            target_2r=target_2r,
            target_3r=target_3r,
            risk_per_share=_safe_float(c.get("risk_per_share") or c.get("long_risk_per_share") or c.get("short_risk_per_share")),
            or_high=_safe_float(c.get("or_high")),
            or_low=_safe_float(c.get("or_low")),
            vwap_last=_safe_float(c.get("vwap_last")),
            scan_price=_safe_float(c.get("price") if c.get("price") is not None else c.get("last_price")),
            scan_seed_score=seed,
            combined_score=_safe_float(c.get("combined_score") if c.get("combined_score") is not None else c.get("score")),
            ml_score=_safe_float(c.get("ml_score") if c.get("ml_score") is not None else c.get("ml")),
            model_bucket=(str(c.get("model_bucket")).strip() if c.get("model_bucket") is not None else None),
            confidence_score=_safe_float(c.get("confidence_score")),
            confidence_grade=(str(c.get("confidence_grade")).strip().upper() if c.get("confidence_grade") is not None else None),
            regime_profile=(str(c.get("regime_profile")).strip() if c.get("regime_profile") is not None else None),
            # Only use ml_score as p_2r_30m proxy if it carries meaningful signal (>= 0.10).
            # A near-zero ml_score (e.g. diagnostic-only profile) must not produce a PASS decision;
            # it should produce WAIT (via p_2r_30m=None → ml_unavailable path).
            p_2r_30m=_safe_float(c.get("p_2r_30m")) if c.get("p_2r_30m") is not None else (
                _safe_float(c.get("ml_score")) if (_safe_float(c.get("ml_score")) or 0.0) >= 0.10 else None
            ),
            probability_source=("entry_now" if c.get("p_2r_30m") is not None else (
                "scan_ml_score" if ((_safe_float(c.get("ml_score")) or 0.0) >= 0.10) else None
            )),
            gate_passes=(bool(c.get("gate_passes")) if c.get("gate_passes") is not None else None),
            gate_fail_reasons=list(c.get("gate_fail_reasons") or []) if isinstance(c.get("gate_fail_reasons"), (list, tuple)) else [],
            monitor_seed=(bool(c.get("monitor_seed")) if c.get("monitor_seed") is not None else None),
            monitor_seed_reasons=list(c.get("monitor_seed_reasons") or []) if isinstance(c.get("monitor_seed_reasons"), (list, tuple)) else [],
            orb_retest_ready=(bool(c.get("orb_retest_ready")) if c.get("orb_retest_ready") is not None else None),
            tradable_now=(bool(c.get("tradable_now")) if c.get("tradable_now") is not None else None),
            trade_ready_passes=(bool(c.get("trade_ready_passes")) if c.get("trade_ready_passes") is not None else None),
            catalyst_score=_safe_float(c.get("catalyst_score")),
            catalyst_confidence=_safe_float(c.get("catalyst_confidence")),
            catalyst_article_count=_safe_int(c.get("catalyst_article_count")),
            catalyst_freshness_hours=_safe_float(c.get("catalyst_freshness_hours")),
            catalyst_tags=list(c.get("catalyst_tags") or []) if isinstance(c.get("catalyst_tags"), (list, tuple)) else [],
            catalyst_headlines=list(c.get("top_headlines") or []) if isinstance(c.get("top_headlines"), (list, tuple)) else [],
            promoted_by_news=bool(c.get("promoted_by_news")),
        )

    def _assign_playbook(self, c: dict[str, Any]) -> str:
        strategy = str(c.get("strategy") or "").strip().lower()
        exec_style = str(c.get("exec_style") or c.get("execution") or "").strip().lower()
        tags = [str(x).lower() for x in (c.get("catalyst_tags") or [])]
        if strategy in {"rr", "range_reversion"}:
            return "range_reversion"
        if c.get("promoted_by_news") or tags:
            return "catalyst_news_ignition"
        if exec_style == "retest":
            return "retest_reclaim_orb"
        return "open_drive_orb"

    

    def _build_seed_from_provider(self, symbol: str, *, playbook: str, seed_source: str = "manual") -> MonitorSymbolState:
        if self._provider is None:
            raise RuntimeError("provider_not_configured")
        sym = str(symbol or "").strip().upper()
        if not sym:
            raise RuntimeError("symbol_required")

        session_date = datetime.now(timezone.utc).astimezone(ET).date()
        bars = self._provider.get_bars_range(
            symbol=sym,
            interval="1m",
            from_d=session_date,
            to_d=session_date,
            include_prepost=False,
        )
        if bars is None or bars.empty:
            raise RuntimeError(f"seed_build_no_intraday_bars:{sym}")

        cols = {str(c).lower(): c for c in bars.columns}
        high_col = cols.get("high")
        low_col = cols.get("low")
        close_col = cols.get("close")
        vol_col = cols.get("volume")
        if not high_col or not low_col or not close_col or not vol_col:
            raise RuntimeError(f"seed_build_missing_columns:{sym}")

        first5 = bars.iloc[:5]
        if first5.empty or len(first5) < 3:
            raise RuntimeError(f"seed_build_no_opening_range:{sym}")

        snapshots = self._provider.get_snapshots([sym])
        snap = snapshots.get(sym) or {}
        reference_price = _safe_float(snap.get("reference_price"))
        if reference_price is None:
            trade = snap.get("latest_trade") or {}
            reference_price = _safe_float(trade.get("price"))
        if reference_price is None:
            minute_bar = snap.get("minute_bar") or {}
            reference_price = _safe_float(minute_bar.get("close"))
        if reference_price is None:
            raise RuntimeError(f"seed_build_snapshot_missing_price:{sym}")

        or_high = float(first5[high_col].astype(float).max())
        or_low = float(first5[low_col].astype(float).min())
        closes = bars[close_col].astype(float)
        vols = bars[vol_col].astype(float)
        vwap_last = float((closes * vols).cumsum().iloc[-1] / max(1e-9, vols.cumsum().iloc[-1]))
        above_vwap = reference_price >= vwap_last
        side = "long" if above_vwap else "short"
        if side == "long":
            entry = or_high
            stop = or_low
        else:
            entry = or_low
            stop = or_high
        risk = max(abs(entry - stop), 1e-9)
        return MonitorSymbolState(
            symbol=sym,
            playbook=playbook,
            seed_source=seed_source,
            best_side=side,
            entry=entry,
            stop_loss=stop,
            target_2r=(entry + 2 * risk) if side == "long" else (entry - 2 * risk),
            target_3r=(entry + 3 * risk) if side == "long" else (entry - 3 * risk),
            risk_per_share=risk,
            or_high=or_high,
            or_low=or_low,
            vwap_last=vwap_last,
            scan_price=reference_price,
            scan_seed_score=50.0,
            combined_score=50.0,
            confidence_score=50.0,
            confidence_grade="C",
            p_2r_30m=0.50,
            probability_source="manual_seed_default",
            gate_passes=True,
            gate_fail_reasons=[],
            monitor_seed=True,
            monitor_seed_reasons=["manual_seed"],
            orb_retest_ready=True,
            tradable_now=bool((side == "long" and reference_price >= entry) or (side == "short" and reference_price <= entry)),
            trade_ready_passes=True,
        )



    def _retire_running_sessions_locked(self, *, watch_modes: set[str] | None = None, source: str | None = None) -> None:
        now = _now_ts()
        for sess in self._sessions.values():
            if not sess.running:
                continue
            if watch_modes is not None and sess.watch_mode not in watch_modes:
                continue
            if source is not None and sess.source != source:
                continue
            sess.running = False
            sess.updated_at = now
            sess.symbols = {}
            sess.seed_symbols = []
            sess.promotion_candidates = []
            self._persist_session(sess)

    @staticmethod
    def _stop_hit_now(st: MonitorSymbolState) -> tuple[bool, str | None]:
        """Check if the stop loss has been hit using last-trade price only.

        We intentionally exclude bid/ask from stop detection. Wide-spread names
        (especially sub-$5 momentum stocks) routinely have asks far above the stop
        level even when the actual trade price never got close — including during
        pre-market and session-closed periods. Using bid/ask produced a flood of
        phantom stopped_out signals (validated against real intraday data 2026-04-14).
        Last-trade price is the authoritative fill reference for a stop order.
        """
        side = (st.best_side or "").strip().lower()
        stop = _safe_float(st.stop_loss)
        if stop is None or side not in {"long", "short"} or st.price is None:
            return False, None
        price = float(st.price)
        if side == "long" and price <= float(stop):
            return True, "stopped_out"
        if side == "short" and price >= float(stop):
            return True, "stopped_out"
        return False, None

    def start_from_scan_candidates(
        self,
        *,
        job_id: str,
        candidates: List[dict],
        top_n: int = 10,
        feed: str = "sip",
        provider: Any = None,
        stream_cache: Any = None,
        symbols_order: Optional[List[str]] = None,
        long_only: bool = False,
        short_only: bool = False,
        source: str = "scan_job_top_n",
        promotion_candidates: Optional[List[str]] = None,
    ) -> MonitorSession:
        if provider is not None and self._provider is None:
            self.configure_runtime(provider=provider, stream_cache=stream_cache, store=self._store, context_engine=self._context_engine, pubsub=self._pubsub)
        if stream_cache is not None and self._stream_cache is None:
            self.configure_runtime(provider=(provider or self._provider), stream_cache=stream_cache, store=self._store, context_engine=self._context_engine, pubsub=self._pubsub)
        if self._provider is None and self._stream_cache is None:
            raise ValueError("provider_or_stream_cache_required")

        ranked = [c for c in (candidates or []) if isinstance(c, dict) and str(c.get("symbol") or "").strip()]
        ranked.sort(key=_best_seed_score, reverse=True)
        picked: list[dict[str, Any]] = []
        if symbols_order:
            order = [str(s).strip().upper() for s in symbols_order if str(s).strip()]
            by_sym = {str(c.get("symbol")).upper(): c for c in ranked}
            for sym in order:
                if sym in by_sym:
                    picked.append(by_sym[sym])
        if not picked:
            picked = ranked[: max(1, min(int(top_n or 10), 100))]
        if long_only:
            picked = [c for c in picked if str(c.get("best_side") or "").strip().lower() != "short"]
        if short_only:
            picked = [c for c in picked if str(c.get("best_side") or "").strip().lower() == "short"]
        if not picked:
            raise ValueError("scan_job_has_no_candidates")

        now = _now_ts()
        with self._lock:
            self._retire_running_sessions_locked(watch_modes={"scanner", "watchlist"})
        sess = MonitorSession(
            monitor_id=uuid.uuid4().hex,
            job_id=str(job_id or "").strip() or None,
            feed_requested=(feed or "sip").strip().lower() or "sip",
            feed_used=(feed or "sip").strip().lower() or "sip",
            source=source,
            mode="scanner",
            started_at=now,
            updated_at=now,
            long_only=long_only,
            short_only=short_only,
            watch_mode="scanner",
            promotion_candidates=[str(s).strip().upper() for s in (promotion_candidates or []) if str(s).strip()],
        )
        for idx, c in enumerate(picked, start=1):
            playbook = self._assign_playbook(c)
            st = self._make_seed_state(c, idx=idx, playbook=playbook, seed_source="scan")
            if not st.symbol:
                continue
            sess.symbols[st.symbol] = st
            sess.seed_symbols.append(st.symbol)

        if not sess.symbols:
            raise ValueError("no_valid_symbols_in_candidates")

        self._promote_news(sess, source_symbols=list(sess.promotion_candidates[:150]), now_ts=now)

        with self._lock:
            for _existing in list(self._sessions.values()):
                if _existing.running:
                    _existing.running = False
                    _existing.updated_at = _now_ts()
                    self._persist_session(_existing)
            self._sessions[sess.monitor_id] = sess
        self._ensure_worker()
        self.refresh(monitor_id=sess.monitor_id, provider=self._provider, stream_cache=self._stream_cache, force=True)
        return sess

    def start_from_symbols(
        self,
        *,
        symbols: List[str],
        feed: str = "sip",
        provider: Any = None,
        stream_cache: Any = None,
    ) -> MonitorSession:
        if provider is not None and self._provider is None:
            self.configure_runtime(provider=provider, stream_cache=stream_cache, store=self._store, context_engine=self._context_engine, pubsub=self._pubsub)
        if stream_cache is not None and self._stream_cache is None:
            self.configure_runtime(provider=(provider or self._provider), stream_cache=stream_cache, store=self._store, context_engine=self._context_engine, pubsub=self._pubsub)
        if self._provider is None and self._stream_cache is None:
            raise ValueError("provider_or_stream_cache_required")
        syms = [str(s).strip().upper() for s in (symbols or []) if str(s).strip()]
        if not syms:
            raise ValueError("symbols_required")
        syms = list(dict.fromkeys(syms))[:100]
        now = _now_ts()
        sess = MonitorSession(
            monitor_id=uuid.uuid4().hex,
            job_id=None,
            feed_requested=(feed or "sip").strip().lower() or "sip",
            feed_used=(feed or "sip").strip().lower() or "sip",
            source="manual_symbols",
            mode="watchlist",
            started_at=now,
            updated_at=now,
            watch_mode="watchlist",
        )
        for sym in syms:
            try:
                st = self._build_seed_from_provider(sym, playbook="open_drive_orb", seed_source="manual")
            except Exception as e:
                sess.seed_errors[sym] = str(e)
                continue
            sess.symbols[sym] = st
            sess.seed_symbols.append(sym)
        if not sess.symbols:
            raise RuntimeError(f"no_symbols_seeded — all failed: {sess.seed_errors}")
        with self._lock:
            for _existing in list(self._sessions.values()):
                if _existing.running:
                    _existing.running = False
                    _existing.updated_at = _now_ts()
                    self._persist_session(_existing)
            self._sessions[sess.monitor_id] = sess
        self._ensure_worker()
        self.refresh(monitor_id=sess.monitor_id, provider=self._provider, stream_cache=self._stream_cache, force=True)
        return sess

    def _build_seed_from_watchlist_plan(self, plan: dict[str, Any]) -> MonitorSymbolState:
        """
        Seed a MonitorSymbolState directly from a desk watchlist plan dict.
        Uses the trader's pre-defined entry/stop/target instead of OR-derived levels.
        """
        sym = str(plan.get("symbol") or "").strip().upper()
        if not sym:
            raise RuntimeError("plan_missing_symbol")

        side = str(plan.get("side") or "").strip().lower()
        if side not in ("long", "short"):
            raise RuntimeError(f"plan_invalid_side:{sym}")

        entry = _safe_float(plan.get("trigger_price"))
        stop  = _safe_float(plan.get("stop_price"))
        if entry is None:
            raise RuntimeError(f"plan_missing_trigger_price:{sym}")
        if stop is None:
            raise RuntimeError(f"plan_missing_stop_price:{sym}")

        risk = abs(entry - stop)
        if risk < 1e-6:
            raise RuntimeError(f"plan_zero_risk:{sym}")

        # Use trader's target if set, else compute 2R
        target = _safe_float(plan.get("target_price"))
        if target is None:
            target = (entry + 2.0 * risk) if side == "long" else (entry - 2.0 * risk)
        target_3r = (entry + 3.0 * risk) if side == "long" else (entry - 3.0 * risk)

        # Best-effort current price for scan_price reference
        scan_price = entry
        if self._provider is not None:
            try:
                snaps = self._provider.get_snapshots([sym])
                snap  = (snaps or {}).get(sym) or {}
                ref   = _safe_float(snap.get("reference_price"))
                if ref is None:
                    trade = snap.get("latest_trade") or {}
                    ref   = _safe_float(trade.get("price"))
                if ref is None:
                    mb = snap.get("minute_bar") or {}
                    ref = _safe_float(mb.get("close"))
                if ref is not None:
                    scan_price = ref
            except Exception:
                pass

        return MonitorSymbolState(
            symbol=sym,
            playbook="watchlist_plan",
            seed_source="watchlist",
            best_side=side,
            entry=entry,
            stop_loss=stop,
            target_2r=target,
            target_3r=target_3r,
            risk_per_share=risk,
            or_high=None,
            or_low=None,
            vwap_last=None,
            scan_price=scan_price,
            scan_seed_score=50.0,
            combined_score=50.0,
            confidence_score=50.0,
            confidence_grade="C",
            p_2r_30m=None,
            probability_source="watchlist_plan",
            gate_passes=True,
            gate_fail_reasons=[],
            monitor_seed=True,
            monitor_seed_reasons=["watchlist_plan"],
            tradable_now=True,
            trade_ready_passes=True,
        )

    def _build_seed_from_snapshot(self, sym: str, *, side: str | None = None) -> MonitorSymbolState:
        """
        Minimal seed using only a snapshot price — no intraday bars required.
        Used pre-market or after-hours when no OR data exists yet.
        Symbol loads in the monitor and tracks live prices in 'watch' state.
        side: pass the watchlist plan's side so longs don't get mislabeled as shorts.
        """
        if self._provider is None:
            raise RuntimeError("provider_not_configured")
        snaps = self._provider.get_snapshots([sym])
        snap  = (snaps or {}).get(sym) or {}
        ref   = _safe_float(snap.get("reference_price"))
        if ref is None:
            trade = snap.get("latest_trade") or {}
            ref   = _safe_float(trade.get("price"))
        if ref is None:
            mb  = snap.get("minute_bar") or {}
            ref = _safe_float(mb.get("close"))
        if ref is None:
            raise RuntimeError(f"seed_build_snapshot_missing_price:{sym}")
        _side = side.strip().lower() if side and side.strip().lower() in ("long", "short") else None
        return MonitorSymbolState(
            symbol=sym,
            playbook="open_drive_orb",
            seed_source="snapshot_only",
            best_side=_side,
            scan_price=ref,
            scan_seed_score=50.0,
            combined_score=50.0,
            confidence_score=50.0,
            confidence_grade="C",
            gate_passes=False,
            gate_fail_reasons=["no_intraday_bars_yet"],
            monitor_seed=True,
            monitor_seed_reasons=["snapshot_only"],
            tradable_now=False,
            trade_ready_passes=False,
        )

    def start_from_watchlist_plans(
        self,
        *,
        plans: List[dict[str, Any]],
        feed: str = "sip",
        provider: Any = None,
        stream_cache: Any = None,
    ) -> MonitorSession:
        """
        Start a monitor session seeded from desk watchlist plans.
        Each plan must have symbol, side, trigger_price, stop_price.
        target_price is optional (2R computed if absent).
        """
        if provider is not None and self._provider is None:
            self.configure_runtime(provider=provider, stream_cache=stream_cache, store=self._store, context_engine=self._context_engine, pubsub=self._pubsub)
        if stream_cache is not None and self._stream_cache is None:
            self.configure_runtime(provider=(provider or self._provider), stream_cache=stream_cache, store=self._store, context_engine=self._context_engine, pubsub=self._pubsub)

        now = _now_ts()
        sess = MonitorSession(
            monitor_id=uuid.uuid4().hex,
            job_id=None,
            feed_requested=(feed or "sip").strip().lower() or "sip",
            feed_used=(feed or "sip").strip().lower() or "sip",
            source="watchlist_plans",
            mode="watchlist",
            started_at=now,
            updated_at=now,
            watch_mode="watchlist",
        )
        for plan in plans:
            sym = str(plan.get("symbol") or "").strip().upper()
            if not sym:
                continue
            try:
                st = self._build_seed_from_watchlist_plan(plan)
            except RuntimeError as e:
                err = str(e)
                # No pre-defined levels — fall back to OR-based seeding so the
                # symbol still loads in the monitor (shows GO/WAIT/PASS as normal).
                if "plan_missing_trigger_price" in err or "plan_missing_stop_price" in err or "plan_invalid_side" in err:
                    _plan_side = str(plan.get("side") or "").strip().lower() or None
                    try:
                        st = self._build_seed_from_provider(sym, playbook="open_drive_orb", seed_source="watchlist_fallback")
                        # Honour the watchlist side — rebuild entry/stop/target if provider
                        # picked the opposite direction (e.g. stock above VWAP → long seed,
                        # but plan says short).
                        if _plan_side in ("long", "short") and st.best_side != _plan_side:
                            _e = st.or_low if _plan_side == "short" else st.or_high
                            _s = st.or_high if _plan_side == "short" else st.or_low
                            _r = max(abs(_e - _s), 1e-9)
                            st.best_side = _plan_side
                            st.entry = _e
                            st.stop_loss = _s
                            st.risk_per_share = _r
                            st.target_2r = (_e - 2 * _r) if _plan_side == "short" else (_e + 2 * _r)
                            st.target_3r = (_e - 3 * _r) if _plan_side == "short" else (_e + 3 * _r)
                            _ref = st.scan_price or 0.0
                            st.tradable_now = bool(
                                (_plan_side == "long" and _ref >= _e) or
                                (_plan_side == "short" and _ref <= _e)
                            )
                        elif _plan_side in ("long", "short"):
                            st.best_side = _plan_side
                    except Exception as e2:
                        # Pre-market / after-hours — no OR bars yet. Seed from snapshot
                        # so the symbol still loads and tracks live price in watch state.
                        if "seed_build_no_intraday_bars" in str(e2) and self._provider is not None:
                            try:
                                st = self._build_seed_from_snapshot(sym, side=_plan_side)
                            except Exception as e3:
                                sess.seed_errors[sym] = str(e3)
                                continue
                        else:
                            sess.seed_errors[sym] = str(e2)
                            continue
                else:
                    sess.seed_errors[sym] = err
                    continue
            except Exception as e:
                sess.seed_errors[sym] = str(e)
                continue
            # Apply scan_score from notes if present (written by premarket_watchlist.py)
            _notes = str(plan.get("notes") or "")
            if "scan_score=" in _notes:
                try:
                    import re as _re
                    _m = _re.search(r"scan_score=([0-9.]+)", _notes)
                    if _m:
                        _ss = float(_m.group(1))
                        if 0.0 < _ss <= 1.0:
                            st.scan_seed_score = round(_ss * 100.0, 1)
                            st.combined_score = _ss
                            if st.ml_score is None:
                                st.ml_score = _ss
                except Exception:
                    pass
            sess.symbols[sym] = st
            sess.seed_symbols.append(sym)

        if not sess.symbols:
            raise RuntimeError(f"no_watchlist_symbols_seeded — all failed: {sess.seed_errors}")

        with self._lock:
            for _existing in list(self._sessions.values()):
                if _existing.running:
                    _existing.running = False
                    _existing.updated_at = _now_ts()
                    self._persist_session(_existing)
            self._sessions[sess.monitor_id] = sess
        self._ensure_worker()
        self.refresh(monitor_id=sess.monitor_id, provider=self._provider, stream_cache=self._stream_cache, force=True)
        return sess

    def active_session(self, *, watch_modes: set[str] | None = None) -> MonitorSession | None:
        with self._lock:
            running = [s for s in self._sessions.values() if s.running]
            if watch_modes is not None:
                running = [s for s in running if s.watch_mode in watch_modes]
            if not running:
                return None
            running.sort(key=lambda s: float(s.updated_at or s.started_at or 0.0), reverse=True)
            return running[0]

    def stop(self, monitor_id: str) -> bool:
        mid = str(monitor_id or "").strip()
        if not mid:
            return False
        with self._lock:
            sess = self._sessions.get(mid)
            if not sess:
                return False
            sess.running = False
            sess.updated_at = _now_ts()
            self._persist_session(sess)
            return True

    def status(self, monitor_id: str, *, provider: Any = None, stream_cache: Any = None, refresh: bool = True) -> dict[str, Any]:
        if refresh:
            self.refresh(monitor_id=monitor_id, provider=(provider or self._provider), stream_cache=(stream_cache or self._stream_cache), force=False)
        with self._lock:
            sess = self._sessions.get(str(monitor_id or "").strip())
            if sess is None:
                raise KeyError("unknown_monitor_id")
            return self._session_to_api(sess)

    def recent_alerts(self, *, monitor_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if monitor_id:
                sess = self._sessions.get(str(monitor_id or "").strip())
                if sess:
                    return [a.to_api() for a in sorted(sess.alerts, key=lambda x: x.event_ts, reverse=True)[:limit]]
        if self._store is not None:
            return self._store.recent_alerts(monitor_id=monitor_id, limit=limit)
        return []

    def replay(self, *, monitor_id: str) -> dict[str, Any]:
        with self._lock:
            sess = self._sessions.get(str(monitor_id or "").strip())
            if sess is not None:
                payload = self._session_to_api(sess)
                return {"ok": True, "replay": {"session": payload, "alerts": [a.to_api() for a in sess.alerts]}}
        if self._store is not None:
            return {"ok": True, "replay": self._store.replay_session(monitor_id)}
        raise KeyError("unknown_monitor_id")

    def _maybe_score_ml_for_watchlist_symbols(self, monitor_id: str, *, provider: Any, now_ts: float) -> None:
        """Score the ORB ML model for watchlist_plan symbols that have no ml_score yet.

        Called outside the main lock, after the ORB window has had time to form (≥9:35 ET).
        Each symbol is scored at most once per session (ml_score_attempted flag prevents re-queuing).
        Writes results back under a brief re-acquisition of the lock.
        """
        # Only run after at least 1 minute of regular session bars exist (9:31+).
        # The ML model needs intraday 1m bars; 9:30 is the opening bar which may not
        # be persisted in Alpaca's API yet when the first refresh fires.
        et_now = datetime.fromtimestamp(now_ts, tz=ET)
        orb_ready = (et_now.hour == 9 and et_now.minute >= 31) or et_now.hour >= 10
        if not orb_ready:
            return

        # Collect symbols that still need scoring.
        # ml_score_attempted is only set True on SUCCESS — failed attempts are retried
        # on the next refresh cycle until a score is obtained or the session ends.
        _WATCHLIST_SEED_SOURCES = {"watchlist", "watchlist_fallback", "snapshot_only"}
        candidates: list[tuple[str, float]] = []
        with self._lock:
            sess = self._sessions.get(str(monitor_id))
            if sess is None:
                return
            for st in sess.symbols.values():
                if (
                    st.seed_source in _WATCHLIST_SEED_SOURCES
                    and not st.ml_score_attempted
                    and st.ml_score is None
                ):
                    ref_price = st.price or st.scan_price
                    if ref_price and float(ref_price) > 0:
                        candidates.append((st.symbol, float(ref_price)))

        if not candidates:
            return

        from ml.orb_model_service import score_orb_symbol

        scored: dict[str, dict[str, Any]] = {}
        for sym, price in candidates:
            try:
                result = score_orb_symbol(sym, last_price=price, provider=provider)
                scored[sym] = result
            except Exception as e:
                scored[sym] = {"score": None, "bucket": None, "error": str(e)}

        # Write results back
        with self._lock:
            sess = self._sessions.get(str(monitor_id))
            if sess is None:
                return
            for sym, result in scored.items():
                st = sess.symbols.get(sym)
                if st is None:
                    continue
                score = result.get("score")
                if score is not None:
                    st.ml_score = float(score)
                    st.combined_score = float(score)  # mirrors scanner: combined_score = ml_score
                    st.model_bucket = result.get("bucket")
                    # Promote to p_2r_30m proxy if meaningful and not already set
                    if st.p_2r_30m is None and float(score) >= 0.10:
                        st.p_2r_30m = float(score)
                        st.probability_source = "live_ml_score"

    def _maybe_upgrade_snapshot_seeds(self, monitor_id: str, *, provider: Any, now_ts: float) -> None:
        """
        Once intraday bars are available (9:31+ ET), upgrade snapshot-only seeds to full
        OR-based seeds so that entry/stop/target/vwap are properly populated.

        Snapshot seeds are created pre-market when no bars exist yet. They track price
        in a 'watch' state but have no OR levels. This method runs on each refresh cycle
        and upgrades any remaining snapshot seeds as soon as bars are available.
        It preserves all live state (price, quote, scores, history) from the old seed.
        """
        et_now = datetime.fromtimestamp(now_ts, tz=ET)
        orb_ready = (et_now.hour == 9 and et_now.minute >= 31) or et_now.hour >= 10
        if not orb_ready:
            return

        # Collect snapshot-seeded symbols that still lack OR levels (entry is None)
        candidates: list[tuple[str, str | None]] = []
        with self._lock:
            sess = self._sessions.get(str(monitor_id))
            if sess is None:
                return
            for st in sess.symbols.values():
                if st.seed_source == "snapshot_only" and st.entry is None:
                    candidates.append((st.symbol, st.best_side))

        # Build full OR seeds — network calls run outside the lock
        # (even when candidates is empty we still need to fill watchlist OR/VWAP below)
        upgraded: dict[str, MonitorSymbolState] = {}
        for sym, plan_side in candidates:
            try:
                new_st = self._build_seed_from_provider(sym, playbook="open_drive_orb", seed_source="watchlist_fallback")
                # Honour the watchlist plan side — rebuild entry/stop/target if provider
                # picked the opposite direction via VWAP detection.
                if plan_side in ("long", "short") and new_st.best_side != plan_side:
                    _e = new_st.or_low if plan_side == "short" else new_st.or_high
                    _s = new_st.or_high if plan_side == "short" else new_st.or_low
                    _r = max(abs(_e - _s), 1e-9)
                    new_st.best_side = plan_side
                    new_st.entry = _e
                    new_st.stop_loss = _s
                    new_st.risk_per_share = _r
                    new_st.target_2r = (_e - 2 * _r) if plan_side == "short" else (_e + 2 * _r)
                    new_st.target_3r = (_e - 3 * _r) if plan_side == "short" else (_e + 3 * _r)
                elif plan_side in ("long", "short"):
                    new_st.best_side = plan_side
                upgraded[sym] = new_st
            except Exception:
                pass  # Bars not available yet — retry on next refresh cycle

        # Merge upgraded OR seed data into the existing live state, preserving live data
        with self._lock:
            sess = self._sessions.get(str(monitor_id))
            if sess is None:
                return
            for sym, new_st in upgraded.items():
                old_st = sess.symbols.get(sym)
                if old_st is None:
                    continue
                old_st.seed_source = new_st.seed_source
                old_st.or_high = new_st.or_high
                old_st.or_low = new_st.or_low
                old_st.vwap_last = new_st.vwap_last
                old_st.entry = new_st.entry
                old_st.stop_loss = new_st.stop_loss
                old_st.target_2r = new_st.target_2r
                old_st.target_3r = new_st.target_3r
                old_st.risk_per_share = new_st.risk_per_share
                old_st.best_side = new_st.best_side
                old_st.tradable_now = new_st.tradable_now
                old_st.gate_passes = new_st.gate_passes
                old_st.orb_retest_ready = new_st.orb_retest_ready
                old_st.monitor_seed_reasons = list(old_st.monitor_seed_reasons) + ["upgraded_from_snapshot"]

        # ── Fill OR + VWAP for watchlist/scan seeds (background thread) ─────
        # These seeds start with or_high=None. We fire a one-shot background
        # thread to fetch bars and fill them so the refresh request path is
        # never blocked by network I/O.
        watchlist_missing: list[str] = []
        with self._lock:
            sess = self._sessions.get(str(monitor_id))
            if sess is not None:
                for st in sess.symbols.values():
                    if st.seed_source in ("watchlist", "scan") and st.or_high is None:
                        # Mark as in-progress immediately to prevent duplicate threads
                        st.or_high = -1.0
                        watchlist_missing.append(st.symbol)

        if watchlist_missing and self._provider is not None:
            import threading as _threading

            def _fill_or_vwap(mid: str, symbols: list[str]) -> None:
                from scanner.indicators import vwap as _vwap
                session_date = datetime.now(timezone.utc).astimezone(ET).date()
                for sym in symbols:
                    try:
                        bars = self._provider.get_bars_range(
                            symbol=sym,
                            interval="1m",
                            from_d=session_date,
                            to_d=session_date,
                            include_prepost=False,
                            timeout_s=8,
                        )
                        if bars is None or bars.empty or len(bars) < 5:
                            # Reset sentinel so we retry next cycle
                            with self._lock:
                                s2 = (self._sessions.get(mid) or MonitorSession.__new__(MonitorSession)).symbols.get(sym)
                                if s2 is not None and s2.or_high == -1.0:
                                    s2.or_high = None
                            continue
                        bars = bars.sort_index()
                        first5 = bars.iloc[:5]
                        cols = {str(c).lower(): c for c in bars.columns}
                        high_col = cols.get("high")
                        low_col  = cols.get("low")
                        if not high_col or not low_col:
                            continue
                        or_high = float(first5[high_col].astype(float).max())
                        or_low  = float(first5[low_col].astype(float).min())
                        vw = _vwap(bars)
                        vwap_val = float(vw.iloc[-1]) if vw is not None and len(vw) > 0 else None
                        with self._lock:
                            sess2 = self._sessions.get(mid)
                            if sess2 is None:
                                continue
                            st2 = sess2.symbols.get(sym)
                            if st2 is None or st2.seed_source not in ("watchlist", "scan"):
                                continue
                            st2.or_high = or_high
                            st2.or_low  = or_low
                            if vwap_val is not None:
                                st2.vwap_last = vwap_val
                    except Exception as _e:
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "watchlist OR/VWAP fill failed for %s: %r", sym, _e
                        )
                        with self._lock:
                            sess3 = self._sessions.get(mid)
                            if sess3 is not None:
                                st3 = sess3.symbols.get(sym)
                                if st3 is not None and st3.or_high == -1.0:
                                    st3.or_high = None  # reset sentinel so next cycle retries

            _threading.Thread(
                target=_fill_or_vwap,
                args=(str(monitor_id), watchlist_missing),
                daemon=True,
                name=f"or-vwap-fill-{monitor_id}",
            ).start()

    def refresh(self, *, monitor_id: str, provider: Any, stream_cache: Any = None, force: bool = False) -> None:
        if provider is None and stream_cache is None:
            raise ValueError("provider_or_stream_cache_required")
        mid = str(monitor_id or "").strip()

        # Score ML and upgrade snapshot seeds — run before the throttle guard so these
        # fire even when the refresh interval hasn't elapsed.
        if provider is not None:
            self._maybe_upgrade_snapshot_seeds(mid, provider=provider, now_ts=_now_ts())
            self._maybe_score_ml_for_watchlist_symbols(mid, provider=provider, now_ts=_now_ts())

        with self._lock:
            sess = self._sessions.get(mid)
            if sess is None:
                raise KeyError("unknown_monitor_id")
            if not sess.running:
                return
            now = _now_ts()
            if (not force) and (now - float(sess.updated_at or 0.0)) < max(0.2, sess.min_refresh_interval_s):
                return
            symbols = list(sess.symbols.keys())
        if stream_cache is not None:
            try:
                if hasattr(stream_cache, "start"):
                    stream_cache.start()
                if hasattr(stream_cache, "ensure_symbols"):
                    stream_cache.ensure_symbols(symbols)
            except Exception:
                pass
        context = self._context_engine.snapshot() if self._context_engine is not None else {}
        stream_state = None
        if stream_cache is not None:
            try:
                stream_state = stream_cache.state()
            except Exception:
                stream_state = None
        now_ts = _now_ts()
        news_due = False
        with self._lock:
            sess = self._sessions.get(mid)
            if sess is None:
                raise KeyError("unknown_monitor_id")
            news_due = (sess.last_news_refresh_ts is None) or ((now_ts - float(sess.last_news_refresh_ts or 0.0)) >= self._news_interval_s)

        if news_due:
            # Call without holding self._lock — each method acquires the lock only around
            # fast state-mutation steps; network calls (fetch_batch, get_bars_range) run
            # unlocked so a slow/hung external API cannot block monitor_start / monitor_stop.
            self._promote_news(sess, source_symbols=list(sess.promotion_candidates[:150]), now_ts=now_ts)
            self._refresh_catalyst_for_session(sess, now_ts=now_ts)

        # ── Batch snapshot fetch for symbols not served by the stream cache ───
        # Calling get_latest_trade + get_latest_quote individually per symbol is
        # O(N) REST calls. Instead, collect all symbols that miss in the stream,
        # fetch them in one get_snapshots() call, then let _read_live_symbol use
        # the pre-fetched snapshot so no extra REST calls are needed.
        batch_snaps: dict[str, dict] = {}
        if provider is not None:
            stream_miss: list[str] = []
            for sym in symbols:
                try:
                    latest_trade_payload(stream_cache, sym, max_age_sec=30.0)
                except Exception:
                    stream_miss.append(sym)
            if stream_miss:
                try:
                    raw = provider.get_snapshots(stream_miss) or {}
                    for sym, snap in raw.items():
                        batch_snaps[str(sym).upper()] = snap
                except Exception:
                    pass  # fall back to per-symbol calls below

        updates: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            updates[sym] = self._read_live_symbol(
                sym, provider=provider, stream_cache=stream_cache,
                now_ts=now_ts, snapshot=batch_snaps.get(sym.upper()),
            )

        with self._lock:
            sess = self._sessions.get(mid)
            if sess is None:
                raise KeyError("unknown_monitor_id")
            sess.latest_context = dict(context or {})
            sess.stream_enabled = stream_cache is not None
            sess.stream_connected = bool(getattr(stream_state, "connected", False)) if stream_state is not None else None
            sess.stream_error = getattr(stream_state, "error", None) if stream_state is not None else None

            for sym, upd in updates.items():
                st = sess.symbols.get(sym)
                if st is None:
                    continue
                self._apply_update_locked(sess, st, upd, context=context, now_ts=now_ts)

            sess.updated_at = now_ts
            sess.refresh_count += 1
            self._persist_session(sess)

    def _read_live_symbol(self, sym: str, *, provider: Any, stream_cache: Any, now_ts: float, snapshot: dict | None = None) -> dict[str, Any]:
        upd: dict[str, Any] = {"last_refresh_at": now_ts, "last_error": None}

        try:
            upd.update(_trade_refresh_update(sym, provider=provider, stream_cache=stream_cache, snapshot=snapshot))
        except MonitorTradeRefreshFailure as e:
            upd["last_error"] = failure_string(e)

        try:
            upd.update(_quote_refresh_update(sym, provider=provider, stream_cache=stream_cache, snapshot=snapshot))
        except MonitorQuoteRefreshFailure as e:
            if upd.get("last_error"):
                upd["last_error"] = str(upd["last_error"]) + "|" + failure_string(e)
            else:
                upd["last_error"] = failure_string(e)

        return upd

    def _apply_update_locked(self, sess: MonitorSession, st: MonitorSymbolState, upd: dict[str, Any], *, context: dict[str, Any], now_ts: float) -> None:
        st.price = _safe_float(upd.get("price"))
        st.trade_size = _safe_int(upd.get("trade_size"))
        if st.price is not None:
            st.price_history.append((st.price, now_ts))
            # Keep a 2-minute rolling window, max 60 entries
            cutoff = now_ts - 120.0
            st.price_history = [p for p in st.price_history if p[1] >= cutoff][-60:]
        st.trade_ts = _parse_ts(upd.get("trade_ts"))
        st.bid = _safe_float(upd.get("bid"))
        st.ask = _safe_float(upd.get("ask"))
        st.bid_size = _safe_int(upd.get("bid_size"))
        st.ask_size = _safe_int(upd.get("ask_size"))
        st.quote_ts = _parse_ts(upd.get("quote_ts"))
        st.last_refresh_at = now_ts
        st.last_error = upd.get("last_error")

        st.market_session_state = _session_state(now_ts)
        st.time_of_day_bucket = _time_of_day_bucket(now_ts)
        st.quote_age_ms = _age_ms(st.quote_ts, now_ts)
        st.trade_age_ms = _age_ms(st.trade_ts, now_ts)
        if st.bid is not None and st.ask is not None and st.bid > 0 and st.ask > 0 and st.ask >= st.bid:
            mid = (st.bid + st.ask) / 2.0
            st.spread_pct = ((st.ask - st.bid) / mid) * 100.0 if mid > 0 else None
            # L2 imbalance snapshot: positive = more size on bid = buy pressure
            if st.bid_size is not None and st.ask_size is not None and (st.bid_size + st.ask_size) > 0:
                st.bid_ask_imbalance = (st.bid_size - st.ask_size) / (st.bid_size + st.ask_size)
            # Trade aggressor detection (simplified Lee-Ready)
            if st.price is not None:
                if st.price >= st.ask:
                    aggressor = 1.0    # lifted the offer — aggressive buy
                elif st.price <= st.bid:
                    aggressor = -1.0   # hit the bid — aggressive sell
                else:
                    aggressor = None   # passive / midpoint — skip
                if aggressor is not None:
                    st.aggressor_history.append((aggressor, now_ts))
                    _of_cutoff = now_ts - 60.0
                    st.aggressor_history = [x for x in st.aggressor_history if x[1] >= _of_cutoff][-100:]
            # Order flow score: 60% L2 imbalance + 40% recent aggressor direction (0=full sell, 1=full buy)
            if st.bid_ask_imbalance is not None:
                _bai_norm = (st.bid_ask_imbalance + 1.0) / 2.0  # map [-1,1] → [0,1]
                if st.aggressor_history:
                    _ag_vals = [x[0] for x in st.aggressor_history[-20:]]
                    _ag_norm = (sum(_ag_vals) / len(_ag_vals) + 1.0) / 2.0  # map [-1,1] → [0,1]
                    st.order_flow_score = round(0.60 * _bai_norm + 0.40 * _ag_norm, 4)
                else:
                    st.order_flow_score = round(_bai_norm, 4)
        else:
            st.spread_pct = None

        if st.price is not None and st.vwap_last not in (None, 0.0):
            st.vwap_delta_pct_live = ((float(st.price) - float(st.vwap_last)) / float(st.vwap_last)) * 100.0
            st.above_vwap_live = float(st.price) >= float(st.vwap_last)
        else:
            st.vwap_delta_pct_live = None
            st.above_vwap_live = None

        if st.entry is not None and st.risk_per_share not in (None, 0.0) and st.price is not None:
            rps = max(abs(float(st.risk_per_share)), 1e-9)
            if (st.best_side or "") == "short":
                st.live_chase_r = (float(st.entry) - float(st.price)) / rps
                st.retest_distance_r = (float(st.price) - float(st.entry)) / rps
            else:
                st.live_chase_r = (float(st.price) - float(st.entry)) / rps
                st.retest_distance_r = (float(st.entry) - float(st.price)) / rps
        else:
            st.live_chase_r = None
            st.retest_distance_r = None

        st.long_triggered_live = bool(st.price is not None and st.entry is not None and (st.best_side == "long") and float(st.price) >= float(st.entry))
        st.short_triggered_live = bool(st.price is not None and st.entry is not None and (st.best_side == "short") and float(st.price) <= float(st.entry))
        st.retest_confirmed_live = bool(st.last_entry_touch_ts is not None and abs(float(st.retest_distance_r or 999.0)) <= 0.08)

        # Compute tape freshness BEFORE stop detection so we can gate on it.
        # Previously _compute_tape_locked ran after _stop_hit_now, meaning stops
        # could fire on stale/session-closed tape (e.g. after-hours ask spikes).
        self._compute_tape_locked(sess, st, now_ts=now_ts)
        st.near_trigger_live = _near_trigger_live(st)

        # Only check stop when tape is live AND the position has been entered.
        # This prevents phantom stopped_out signals from:
        #   (a) stale or session-closed tape (ask spikes after hours)
        #   (b) wide-spread names where ask >> stop before any fill occurred
        #   (c) untriggered setups where price has not yet reached entry
        #
        # Institutional-grade confirmation: require 2 consecutive ticks at/through stop
        # AND ≥200 cumulative shares printed there before locking terminal state.
        # Single-print wicks that reverse immediately (HFT noise, fat fingers) are ignored.
        _already_entered = (
            st.long_triggered_live
            or st.short_triggered_live
            or st.monitor_state in {"triggered", "confirmed", "extended"}
        )
        if st.tape_live and _already_entered and not st.terminal_lock:
            stop_hit, stop_reason = self._stop_hit_now(st)
            if stop_hit:
                st.stop_hit_ticks += 1
                st.volume_at_stop += float(st.trade_size or 0)
                # Confirm stop: 2+ consecutive ticks AND ≥200 shares through level
                if st.stop_hit_ticks >= 2 and st.volume_at_stop >= 200:
                    st.terminal_lock = True
                    st.terminal_reason = stop_reason or "stopped_out"
                    st.terminal_state_ts = now_ts
                    st.cooldown_until_ts = None
            else:
                # Price moved back away from stop — reset confirmation counters
                st.stop_hit_ticks = 0
                st.volume_at_stop = 0.0

        self._compute_state_and_alerts_locked(sess, st, context=context, now_ts=now_ts)
        if st.playbook == "watchlist_plan":
            self._maybe_log_plan_outcome_locked(sess, st, now_ts=now_ts)
            self._maybe_log_plan_snapshot_locked(sess, st, now_ts=now_ts)
        self._persist_symbol(sess, st)

    def _compute_tape_locked(self, sess: MonitorSession, st: MonitorSymbolState, *, now_ts: float) -> None:
        flags: list[str] = []
        if st.market_session_state == "closed":
            st.tape_live = False
            st.tape_live_reason = "session_closed"
            return
        if st.quote_age_ms is None or st.trade_age_ms is None:
            st.tape_live = False
            st.tape_live_reason = "missing_live_timestamps"
            sess.add_diagnostic("rejected_no_live_confirmation", {"symbol": st.symbol, "reason": "missing_live_timestamps"})
            return
        if st.quote_age_ms > 2500 or st.trade_age_ms > 2500:
            st.tape_live = False
            st.tape_live_reason = "stale_timing"
            st.tape_deterioration = "stale_timing"
            sess.add_diagnostic("rejected_stale_timing", {"symbol": st.symbol, "quote_age_ms": st.quote_age_ms, "trade_age_ms": st.trade_age_ms})
            return
        # Price-tiered spread limit: sub-$5 names routinely have $0.02-0.04 spreads
        # that exceed a flat 0.75% threshold. 0.75% of $1.50 = $0.01 — one tick.
        _spread_limit = 2.0 if (st.price or 999.0) < 5.0 else 1.5
        if st.spread_pct is not None and st.spread_pct > _spread_limit:
            st.tape_live = False
            st.tape_live_reason = "wide_spread"
            st.tape_deterioration = "wide_spread"
            sess.add_diagnostic("rejected_spread", {"symbol": st.symbol, "spread_pct": st.spread_pct, "spread_limit": _spread_limit})
            return
        st.tape_live = True
        st.tape_live_reason = "live"

    def _compute_state_and_alerts_locked(self, sess: MonitorSession, st: MonitorSymbolState, *, context: dict[str, Any], now_ts: float) -> None:
        playbook = PLAYBOOKS.get(st.playbook, PLAYBOOKS["open_drive_orb"])
        context_score = float(context.get("risk_on_score") or 0.0)
        st.context_score = context_score
        live_state = playbook.compute_live_state(st, context, now_ts)

        if st.terminal_lock:
            new_state, reasons, flags = "failed", [st.terminal_reason or "terminal_lock"], [st.terminal_reason or "terminal_lock"]
        else:
            new_state, reasons, flags = playbook.transition(st, live_state, now_ts)

        # Catalyst freshness is first-class.
        if st.playbook == "catalyst_news_ignition" and (st.catalyst_freshness_hours is None or st.catalyst_freshness_hours > 4.0):
            sess.add_diagnostic("rejected_no_catalyst_freshness", {"symbol": st.symbol, "freshness_hours": st.catalyst_freshness_hours})

        # Context disagreement.
        if st.best_side == "long" and float(context.get("risk_on_score") or 0.0) < 0.0 and new_state in {"arming", "triggered"}:
            flags.append("context_misaligned")
            sess.add_diagnostic("rejected_multi_timeframe_disagreement", {"symbol": st.symbol, "risk_on_score": context.get("risk_on_score")})
        if st.best_side == "short" and float(context.get("risk_on_score") or 0.0) > 0.0 and new_state in {"arming", "triggered"}:
            flags.append("context_misaligned")
            sess.add_diagnostic("rejected_multi_timeframe_disagreement", {"symbol": st.symbol, "risk_on_score": context.get("risk_on_score")})

        # Transition tracking.
        old_state = st.monitor_state
        old_bucket = st.alert_state_bucket or _alert_bucket_for_state(st, old_state)
        st.previous_state = old_state
        st.monitor_state = new_state
        st.just_transitioned = (new_state != old_state)
        new_bucket = _alert_bucket_for_state(st, new_state)
        st.alert_state_bucket = new_bucket
        if st.just_transitioned:
            st.last_transition_ts = now_ts
            st.promoted_by_monitor_transition = new_state in {"arming", "confirmed", "triggered"}
            if st.promoted_by_monitor_transition:
                sess.add_diagnostic("promoted_by_monitor_transition", {"symbol": st.symbol, "to_state": new_state})

        # Cooldowns.
        if new_state in {"failed", "extended"}:
            if st.terminal_lock and new_state == "failed":
                st.cooldown_until_ts = None
            else:
                cd = self._cooldowns.get(st.playbook, 120.0)
                st.cooldown_until_ts = now_ts + cd
            if new_state == "failed":
                sess.add_diagnostic("demoted_by_decay", {"symbol": st.symbol, "state": new_state, "terminal_lock": st.terminal_lock})
        elif new_state == "triggered":
            st.last_trigger_ts = now_ts
            st.cooldown_until_ts = now_ts + self._cooldowns.get(st.playbook, 120.0)

        # Score model.
        st.live_score = round(
            _compute_monitor_live_score(
                st,
                playbook=playbook,
                live_state=live_state,
                context=context,
                context_score=context_score,
                flags=flags,
            ),
            4,
        )

        st.live_confidence_reasons = _monitor_live_confidence_reasons(st, reasons)
        st.risk_flags = _dedupe_keep(flags)
        st.live_confidence_score = max(0.0, min(100.0, 55.0 + st.live_score))
        st.live_confidence_grade = _grade(st.live_confidence_score)

        # For watchlist plans, override confidence score with plan readiness
        if st.playbook == "watchlist_plan":
            # Inject session archetype and sector concentration into state before scoring
            st.session_archetype = str(context.get("session_archetype") or "mixed")
            sym_sector = TICKER_SECTOR.get(st.symbol, None)
            if sym_sector:
                st.same_sector_count = sum(
                    1 for other_sym, other_st in sess.symbols.items()
                    if other_sym != st.symbol
                    and other_st.playbook == "watchlist_plan"
                    and other_st.monitor_state not in {"failed"}
                    and TICKER_SECTOR.get(other_sym) == sym_sector
                )
            else:
                st.same_sector_count = 0
            try:
                pr_score, pr_breakdown = compute_plan_readiness(st)
                st.plan_readiness_score = pr_score
                st.plan_readiness_grade_str = _plan_readiness_grade(pr_score)
                st.plan_readiness_breakdown = pr_breakdown
                st.live_confidence_score = pr_score
                st.live_confidence_grade = st.plan_readiness_grade_str
                # Map 0-100 plan readiness to live_score range (-20 to +20)
                st.live_score = round((pr_score / 100.0) * 40.0 - 20.0, 4)
            except Exception:
                pass

        if st.playbook == "watchlist_plan":
            # Watchlist plan symbols: skip ORB build_plan_state entirely.
            # Decision is derived from proximity state + plan readiness score.
            _monitor_state = st.monitor_state or "watch"
            _pr_score = st.plan_readiness_score or 0.0
            if _monitor_state == "triggered":
                st.decision = "GO"
                st.decision_notes = ["entry_reached"]
                st.decision_hint = "At your planned entry level"
            elif _monitor_state == "arming":
                st.decision = "WAIT"
                st.decision_notes = ["approaching_entry"]
                st.decision_hint = "Within 0.5R of entry — get ready"
            elif _monitor_state == "failed":
                st.decision = "PASS"
                st.decision_notes = ["stop_crossed"]
                st.decision_hint = "Plan invalidated — stop crossed"
            elif _pr_score >= 75:
                st.decision = "GO"
                st.decision_notes = ["plan_readiness_high"]
                st.decision_hint = f"Plan Readiness {st.plan_readiness_grade_str} {round(_pr_score)} — conditions favorable"
            elif _pr_score >= 50:
                st.decision = "WAIT"
                st.decision_notes = ["plan_readiness_moderate"]
                st.decision_hint = f"Plan Readiness {st.plan_readiness_grade_str} {round(_pr_score)} — conditions building"
            else:
                st.decision = "PASS"
                st.decision_notes = ["plan_readiness_low"]
                st.decision_hint = f"Plan Readiness {st.plan_readiness_grade_str} {round(_pr_score)} — conditions not aligned"
            st.rejection_reasons = []
            st.diagnostics = {
                "plan_readiness_score": _pr_score,
                "plan_readiness_grade": st.plan_readiness_grade_str,
                "plan_readiness_breakdown": st.plan_readiness_breakdown,
                "monitor_state": _monitor_state,
                "decision": st.decision,
                "decision_notes": list(st.decision_notes),
            }
        else:
            decision, decision_notes, decision_hint = build_plan_state(
                side=(st.best_side or "long"),
                last_price=st.price,
                entry=st.entry,
                stop=st.stop_loss,
                target_2r=st.target_2r,
                p_2r_30m=st.p_2r_30m,
                chase_r=st.live_chase_r,
                vwap_delta_pct=st.vwap_delta_pct_live,
                trend_state=("up" if st.above_vwap_live else ("down" if st.above_vwap_live is False else None)),
            )
            rejection_reasons = list(st.gate_fail_reasons or [])
            if st.monitor_seed_reasons:
                rejection_reasons.extend([f"seed:{x}" for x in st.monitor_seed_reasons])
            if st.tape_live_reason and st.tape_live_reason != "live":
                rejection_reasons.append(st.tape_live_reason)
            rejection_reasons.extend(st.risk_flags)
            if not bool(st.orb_retest_ready):
                rejection_reasons.append("orb_retest_not_ready")
            if st.monitor_state in {"failed", "extended"}:
                rejection_reasons.append(f"monitor_state:{st.monitor_state}")
            st.decision = "GO" if (decision == "GO" and st.monitor_state in {"arming", "confirmed", "triggered"} and st.tape_live) else ("PASS" if decision == "PASS" else "WAIT")
            st.decision_notes = _dedupe_keep(list(decision_notes) + ([f"monitor_state:{st.monitor_state}"] if st.monitor_state else []))
            st.decision_hint = decision_hint
            st.rejection_reasons = _dedupe_keep(rejection_reasons)
            st.diagnostics = {
                "quote_age_ms": st.quote_age_ms,
                "trade_age_ms": st.trade_age_ms,
                "spread_pct": st.spread_pct,
                "tape_live_reason": st.tape_live_reason,
                "context_score": st.context_score,
                "retest_distance_r": st.retest_distance_r,
                "catalyst_freshness_hours": st.catalyst_freshness_hours,
                "gate_fail_reasons": list(st.gate_fail_reasons),
                "rejection_reasons": list(st.rejection_reasons),
                "decision": st.decision,
                "decision_notes": list(st.decision_notes),
            }

        should_emit = False
        if new_bucket != old_bucket and new_bucket in {"near_trigger", "ready", "triggered"}:
            should_emit = True
        elif st.just_transitioned and new_state in {"failed", "extended"} and old_bucket in {"near_trigger", "ready", "triggered"}:
            should_emit = True
        elif st.just_transitioned and new_state == "cooldown" and old_bucket in {"ready", "triggered"}:
            should_emit = True
        if should_emit:
            self._emit_alert_locked(
                sess,
                st,
                old_state,
                new_state,
                reasons=st.live_confidence_reasons,
                flags=st.risk_flags,
                now_ts=now_ts,
                alert_bucket=new_bucket,
            )

    def _dedupe_key(self, sess: MonitorSession, st: MonitorSymbolState, to_state: str, now_ts: float, alert_bucket: str) -> str:
        minute = int(now_ts // 60)
        raw = f"{sess.monitor_id}:{st.symbol}:{st.playbook}:{to_state}:{alert_bucket}:{minute}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _emit_alert_locked(self, sess: MonitorSession, st: MonitorSymbolState, from_state: str, to_state: str, *, reasons: list[str], flags: list[str], now_ts: float, alert_bucket: str) -> None:
        dedupe_key = self._dedupe_key(sess, st, to_state, now_ts, alert_bucket)
        for existing in sess.alerts[-50:]:
            if existing.dedupe_key == dedupe_key:
                return
        event_type = alert_bucket
        alert = MonitorAlertEvent(
            event_id=uuid.uuid4().hex,
            monitor_id=sess.monitor_id,
            symbol=st.symbol,
            playbook=st.playbook,
            from_state=from_state,
            to_state=to_state,
            event_type=event_type,
            event_ts=now_ts,
            price=st.price,
            bid=st.bid,
            ask=st.ask,
            spread_pct=st.spread_pct,
            vwap_delta_pct=st.vwap_delta_pct_live,
            live_chase_r=st.live_chase_r,
            catalyst_score=st.catalyst_score,
            context_score=st.context_score,
            dedupe_key=dedupe_key,
            alert_bucket=alert_bucket,
            reasons=list(reasons or []),
            flags=list(flags or []),
        )
        sess.alerts.append(alert)
        sess.alerts = sess.alerts[-500:]
        st.alert_fired_count += 1
        if self._store is not None:
            try:
                self._store.append_alert(alert.to_api())
            except Exception:
                pass
        if self._pubsub is not None:
            try:
                self._pubsub.publish("alerts", alert.to_api())
            except Exception:
                pass

    def _maybe_log_plan_snapshot_locked(self, sess: MonitorSession, st: MonitorSymbolState, *, now_ts: float) -> None:
        """Log a plan_snapshot record when price is within 1.5R of entry (throttled to once per 120s)."""
        if self._store is None or st.price is None or st.entry is None or st.risk_per_share is None:
            return
        if st.last_snapshot_ts is not None and (now_ts - st.last_snapshot_ts) < 120.0:
            return
        risk = max(abs(float(st.risk_per_share)), 1e-9)
        if st.best_side == "long":
            entry_dist_r = (float(st.entry) - float(st.price)) / risk
        else:
            entry_dist_r = (float(st.price) - float(st.entry)) / risk
        if abs(entry_dist_r) > 1.5:
            return  # too far from entry — not worth snapshotting yet
        try:
            session_date = datetime.now(tz=ET).date().isoformat()
            features = extract_plan_snapshot_features(st, session_date=session_date, snapshot_trigger="proximity")
            self._store.log_plan_snapshot(features, monitor_id=sess.monitor_id)
            st.last_snapshot_ts = now_ts
        except Exception:
            pass

    def _maybe_log_plan_outcome_locked(self, sess: MonitorSession, st: MonitorSymbolState, *, now_ts: float) -> None:
        """Label the most recent unlabeled snapshot when target or stop is crossed."""
        if self._store is None or st.price is None:
            return
        price = float(st.price)
        outcome: str | None = None

        if st.target_2r is not None:
            target = float(st.target_2r)
            if (st.best_side == "long"  and price >= target) or \
               (st.best_side == "short" and price <= target):
                outcome = "target_reached"

        if outcome is None and st.stop_loss is not None:
            stop = float(st.stop_loss)
            if (st.best_side == "long"  and price <= stop) or \
               (st.best_side == "short" and price >= stop):
                outcome = "stopped_out"

        if outcome is not None:
            try:
                session_date = datetime.now(tz=ET).date().isoformat()
                self._store.label_plan_outcome(
                    monitor_id=sess.monitor_id,
                    symbol=st.symbol,
                    session_date=session_date,
                    outcome=outcome,
                    price=price,
                )
            except Exception:
                pass

    def _persist_symbol(self, sess: MonitorSession, st: MonitorSymbolState) -> None:
        if self._store is not None:
            try:
                self._store.save_symbol_state(sess.monitor_id, st.to_api())
            except Exception:
                pass

    def _persist_session(self, sess: MonitorSession) -> None:
        if self._store is not None:
            try:
                payload = self._session_to_api(sess)
                self._store.save_session(payload)
                now_ts = _now_ts()
                if (now_ts - float(getattr(sess, 'last_replay_persist_ts', 0.0) or 0.0)) >= 3.0:
                    self._store.save_replay_snapshot(sess.monitor_id, payload)
                    sess.last_replay_persist_ts = now_ts
            except Exception:
                pass

    def _refresh_catalyst_for_session(self, sess: MonitorSession, *, now_ts: float) -> None:
        """Fetch catalyst data and apply to session.
        MUST be called WITHOUT holding self._lock — fetch_batch is a blocking network call."""
        if self._catalyst is None or not sess.symbols:
            return
        # Snapshot symbol keys under a brief lock, then release before the network call
        with self._lock:
            syms = list(sess.symbols.keys())[:150]
        if not syms:
            return
        try:
            bundles = self._catalyst.fetch_batch(syms, per_symbol_limit=6, lookback_hours=24)
        except Exception:
            return
        # Apply results under lock (fast — no network calls here)
        with self._lock:
            for sym, bundle in bundles.items():
                st = sess.symbols.get(sym)
                if st is None:
                    continue
                st.catalyst_score = _safe_float(getattr(bundle, "score", None))
                st.catalyst_confidence = _safe_float(getattr(bundle, "confidence", None))
                st.catalyst_article_count = _safe_int(getattr(bundle, "article_count", None))
                st.catalyst_freshness_hours = _safe_float(getattr(bundle, "freshness_hours", None))
                st.catalyst_tags = list(getattr(bundle, "tags", []) or [])
                st.catalyst_headlines = list(getattr(bundle, "top_headlines", []) or [])
                arts = list(getattr(bundle, "articles", []) or [])
                if arts:
                    newest = getattr(arts[0], "created_at", None)
                    st.latest_news_at = _parse_ts(newest)
                    st.news_source = getattr(arts[0], "source", None)
            sess.last_news_refresh_ts = now_ts
        # Persist news events to the store outside the lock (DB write, not shared state)
        if self._store is not None:
            for sym, bundle in bundles.items():
                for art in list(getattr(bundle, "articles", []) or [])[:3]:
                    news_id = hashlib.sha256(
                        f"{sym}|{getattr(art,'headline',None)}|{getattr(art,'created_at',None)}".encode("utf-8")
                    ).hexdigest()[:32]
                    payload = {
                        "news_id": news_id,
                        "symbol": sym,
                        "headline": getattr(art, "headline", "") or "",
                        "source": getattr(art, "source", None),
                        "url": getattr(art, "url", None),
                        "published_at": getattr(art, "created_at", None),
                        "received_at": now_ts,
                        "sentiment_score": getattr(art, "score", None),
                        "catalyst_score": getattr(bundle, "score", None),
                        "freshness_sec": (float(getattr(bundle, "freshness_hours", 0.0) or 0.0) * 3600.0) if getattr(bundle, "freshness_hours", None) is not None else None,
                        "tags": list(getattr(art, "tags", []) or []),
                    }
                    try:
                        self._store.append_news_event(payload)
                    except Exception:
                        pass

    def _promote_news(self, sess: MonitorSession, *, source_symbols: list[str], now_ts: float) -> None:
        """Promote high-conviction news symbols into the session.
        MUST be called WITHOUT holding self._lock — fetch_batch and _build_seed_from_provider
        are both blocking network calls."""
        if self._catalyst is None or not source_symbols:
            return
        source_symbols = list(dict.fromkeys([str(s).strip().upper() for s in source_symbols if str(s).strip()]))[:150]
        # Network call — outside lock
        try:
            bundles = self._catalyst.fetch_batch(source_symbols, per_symbol_limit=4, lookback_hours=8)
        except Exception:
            return
        # Snapshot current symbols to avoid holding the lock during seed building
        with self._lock:
            existing_syms = set(sess.symbols.keys())
        # Filter and build seeds — network calls (_build_seed_from_provider) outside lock
        candidates: list[tuple[str, Any, float, Any]] = []
        for sym, bundle in bundles.items():
            if sym in existing_syms:
                continue
            score = _safe_float(getattr(bundle, "score", None)) or 0.0
            fresh = _safe_float(getattr(bundle, "freshness_hours", None))
            if fresh is None or fresh > 2.0:
                continue
            if abs(score) < 0.85:
                continue
            try:
                st = self._build_seed_from_provider(sym, playbook="catalyst_news_ignition", seed_source="news")
            except Exception:
                continue
            candidates.append((sym, bundle, score, st))
        if not candidates:
            return
        # Apply results under lock (fast — no network calls)
        with self._lock:
            promoted: list[str] = []
            for sym, bundle, score, st in candidates:
                if sym in sess.symbols:  # re-check after building
                    continue
                fresh = _safe_float(getattr(bundle, "freshness_hours", None))
                st.promoted_by_news = True
                st.catalyst_score = score
                st.catalyst_confidence = _safe_float(getattr(bundle, "confidence", None))
                st.catalyst_article_count = _safe_int(getattr(bundle, "article_count", None))
                st.catalyst_freshness_hours = fresh
                st.catalyst_tags = list(getattr(bundle, "tags", []) or [])
                st.catalyst_headlines = list(getattr(bundle, "top_headlines", []) or [])
                st.scan_seed_score = max(float(st.scan_seed_score or 50.0), 60.0 + abs(score) * 10.0)
                sess.symbols[sym] = st
                promoted.append(sym)
                sess.add_diagnostic("promoted_by_news", {"symbol": sym, "score": score, "freshness_hours": fresh})
            if promoted:
                sess.seed_symbols.extend(promoted)
                sess.seed_symbols = _dedupe_keep(sess.seed_symbols)

    def _session_to_api(self, sess: MonitorSession) -> dict[str, Any]:
        symbols = [st.to_api() for st in sess.symbols.values()]
        symbols.sort(key=lambda x: (float(x.get("live_score") or 0.0), float(x.get("context_score") or 0.0)), reverse=True)
        state_counts: dict[str, int] = {}
        grade_counts: dict[str, int] = {}
        tape_counts = {"live": 0, "stale_or_missing": 0, "session_closed": 0}
        ready_symbols: list[str] = []
        top_live_symbol = None
        for s in symbols:
            state = str(s.get("monitor_state") or "watch")
            grade = str(s.get("live_confidence_grade") or "D")
            state_counts[state] = state_counts.get(state, 0) + 1
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            if s.get("tape_live"):
                tape_counts["live"] += 1
            elif s.get("tape_live_reason") == "session_closed":
                tape_counts["session_closed"] += 1
            else:
                tape_counts["stale_or_missing"] += 1
            if state in {"arming", "confirmed", "triggered"}:
                ready_symbols.append(str(s.get("symbol")))
            if top_live_symbol is None and s.get("tape_live"):
                top_live_symbol = {"symbol": s.get("symbol"), "live_score": s.get("live_score"), "state": s.get("monitor_state")}
        alerts_recent = [a.to_api() for a in sorted(sess.alerts, key=lambda x: x.event_ts, reverse=True)[:50]]
        return {
            "ok": True,
            "monitor_id": sess.monitor_id,
            "job_id": sess.job_id,
            "feed_requested": sess.feed_requested,
            "feed_used": sess.feed_used,
            "source": sess.source,
            "mode": sess.mode,
            "watch_mode": sess.watch_mode,
            "running": sess.running,
            "started_at": sess.started_at,
            "updated_at": sess.updated_at,
            "refresh_count": sess.refresh_count,
            "stream_enabled": sess.stream_enabled,
            "stream_connected": sess.stream_connected,
            "stream_error": sess.stream_error,
            "long_only": sess.long_only,
            "short_only": sess.short_only,
            "summary": {
                "counts": {
                    "monitor_state": dict(state_counts),
                    "live_confidence_grade": dict(grade_counts),
                    "tape_live": dict(tape_counts),
                },
                "state_counts": dict(state_counts),
                "grade_counts": dict(grade_counts),
                "tape": dict(tape_counts),
                "ready_count": len([s for s in symbols if s.get("monitor_state") in {"arming","confirmed"}]),
                "live_triggered_count": len([s for s in symbols if s.get("monitor_state") == "triggered"]),
                "invalid_count": len([s for s in symbols if s.get("monitor_state") in {"invalid","failed"}]),
                "ready_symbols": ready_symbols[:12],
                "top_live_symbol": top_live_symbol,
                "headline": f"Monitor-first session • {len(symbols)} symbols • {len(alerts_recent)} recent transition alerts",
                "diagnostics": dict(sess.diagnostics),
                "failure_samples": dict(sess.failure_samples),
                "context": dict(sess.latest_context or {}),
            },
            "symbols": symbols,
            "alerts_recent": alerts_recent,
            "context": dict(sess.latest_context or {}),
            "seed_errors": dict(sess.seed_errors),
        }
