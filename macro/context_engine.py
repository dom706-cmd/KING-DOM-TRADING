
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import math
import threading
import time

from providers.base import BarsRequest

SECTOR_ETFS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]

def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or abs(b) < 1e-9:
        return None
    return ((a - b) / b) * 100.0

def classify_session_archetype(
    spy_move: float | None,
    qqq_move: float | None,
    breadth_score: float,
    volatility_regime: str,
    cohort_tape_quality: float,
) -> str:
    """
    Classify the current session into a trading archetype based on macro conditions.
    Used to re-weight Plan Readiness scoring components appropriately for the day type.

    Archetypes:
      gap_and_go  — big move, expanding vol, broad sector participation
      trend_day   — sustained directional drift, broad alignment
      chop        — small move, compressed vol, low breadth
      mixed       — default when signals conflict or are insufficient
    """
    abs_spy  = abs(spy_move  or 0.0)
    abs_qqq  = abs(qqq_move  or 0.0)
    avg_move = (abs_spy + abs_qqq) / 2.0
    breadth  = abs(breadth_score or 0.0)
    cohort   = abs(cohort_tape_quality or 0.0)
    expanding = volatility_regime == "expanding"

    if avg_move >= 1.5 and expanding and breadth >= 0.5:
        return "gap_and_go"
    if avg_move >= 0.7 and breadth >= 0.3 and cohort >= 0.35:
        return "trend_day"
    if avg_move < 0.35 and not expanding and breadth < 0.2:
        return "chop"
    return "mixed"


@dataclass
class ContextSnapshot:
    generated_at: str
    spy_trend_state: str
    qqq_trend_state: str
    breadth_score: float
    risk_on_score: float
    volatility_regime: str
    time_of_day_bucket: str
    sector_strength_by_etf: dict[str, float]
    leaders: list[str]
    laggards: list[str]
    cohort_tape_quality: float
    session_archetype: str
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "spy_trend_state": self.spy_trend_state,
            "qqq_trend_state": self.qqq_trend_state,
            "breadth_score": self.breadth_score,
            "risk_on_score": self.risk_on_score,
            "volatility_regime": self.volatility_regime,
            "time_of_day_bucket": self.time_of_day_bucket,
            "sector_strength_by_etf": dict(self.sector_strength_by_etf),
            "leaders": list(self.leaders),
            "laggards": list(self.laggards),
            "cohort_tape_quality": self.cohort_tape_quality,
            "session_archetype": self.session_archetype,
            "errors": list(self.errors),
        }

class MarketContextEngine:
    def __init__(self, refresh_interval_s: float = 10.0) -> None:
        self.refresh_interval_s = max(2.0, float(refresh_interval_s))
        self._lock = threading.RLock()
        self._snapshot = ContextSnapshot(
            generated_at=datetime.now(timezone.utc).isoformat(),
            spy_trend_state="unknown",
            qqq_trend_state="unknown",
            breadth_score=0.0,
            risk_on_score=0.0,
            volatility_regime="unknown",
            time_of_day_bucket="unknown",
            sector_strength_by_etf={},
            leaders=[],
            laggards=[],
            cohort_tape_quality=0.0,
            session_archetype="mixed",
            errors=[],
        )
        self._provider = None
        self._store = None
        self._stop = threading.Event()
        self._thread = None

    def configure(self, provider: Any, store: Any = None) -> None:
        self._provider = provider
        self._store = store

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="market-context", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        th = self._thread
        if th and th.is_alive():
            th.join(timeout=5.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot.to_dict()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._provider is not None:
                    snap = self._compute_snapshot(self._provider)
                    with self._lock:
                        self._snapshot = snap
                    if self._store is not None:
                        try:
                            self._store.append_context_snapshot(snap.to_dict())
                        except Exception:
                            pass
            except Exception as e:
                with self._lock:
                    snap = self._snapshot
                    errs = list(snap.errors)
                    errs.append(f"{type(e).__name__}: {e}")
                    self._snapshot.errors = errs[-10:]
                    self._snapshot.generated_at = datetime.now(timezone.utc).isoformat()
            self._stop.wait(self.refresh_interval_s)

    def _bucket_time_of_day(self, now_et: datetime) -> str:
        hm = now_et.hour * 60 + now_et.minute
        if 570 <= hm < 615:
            return "open_impulse"
        if 615 <= hm < 690:
            return "post_open"
        if 690 <= hm < 870:
            return "midday"
        if 870 <= hm < 960:
            return "late_day"
        if hm < 570:
            return "premarket"
        return "afterhours"

    def _compute_snapshot(self, provider: Any) -> ContextSnapshot:
        now = datetime.now(timezone.utc)
        now_et = now.astimezone()
        errors: list[str] = []
        sector_strength: dict[str, float] = {}

        def _trend_for(sym: str) -> tuple[str, float | None]:
            try:
                bars = provider.get_bars(BarsRequest(symbol=sym, interval="1m", period="1d", include_prepost=False))
                if bars is None or bars.empty:
                    raise RuntimeError(f"{sym}_bars_empty")
                close_col = "Close" if "Close" in bars.columns else "close"
                open_col = "Open" if "Open" in bars.columns else "open"
                closes = bars[close_col].astype(float)
                opens = bars[open_col].astype(float)
                last_price = float(closes.iloc[-1])
                first_price = float(opens.iloc[0])
                move = _pct(last_price, first_price)
                state = "up" if (move or 0.0) > 0.2 else "down" if (move or 0.0) < -0.2 else "flat"
                return state, move
            except Exception as e:
                errors.append(f"{sym}:{type(e).__name__}:{e}")
                return "unknown", None

        spy_state, spy_move = _trend_for("SPY")
        qqq_state, qqq_move = _trend_for("QQQ")
        positive = 0
        negative = 0
        vals = []
        for sym in SECTOR_ETFS:
            state, move = _trend_for(sym)
            if move is not None:
                vals.append(move)
                sector_strength[sym] = float(move)
                if move > 0:
                    positive += 1
                elif move < 0:
                    negative += 1
        breadth_score = 0.0 if not vals else sum(vals) / len(vals)
        leaders = [k for k, _ in sorted(sector_strength.items(), key=lambda kv: kv[1], reverse=True)[:3]]
        laggards = [k for k, _ in sorted(sector_strength.items(), key=lambda kv: kv[1])[:3]]
        risk_on_score = sum(v for v in [spy_move, qqq_move, breadth_score] if v is not None) / max(1, len([v for v in [spy_move, qqq_move, breadth_score] if v is not None]))
        volatility_regime = "expanding" if (abs(spy_move or 0.0) + abs(qqq_move or 0.0)) / 2.0 > 1.2 else "compressed"
        cohort_tape_quality = max(-1.0, min(1.0, ((positive - negative) / max(1, len(SECTOR_ETFS)))))
        session_archetype = classify_session_archetype(
            spy_move=spy_move, qqq_move=qqq_move,
            breadth_score=breadth_score, volatility_regime=volatility_regime,
            cohort_tape_quality=cohort_tape_quality,
        )
        return ContextSnapshot(
            generated_at=now.isoformat(),
            spy_trend_state=spy_state,
            qqq_trend_state=qqq_state,
            breadth_score=round(breadth_score, 4),
            risk_on_score=round(risk_on_score, 4),
            volatility_regime=volatility_regime,
            time_of_day_bucket=self._bucket_time_of_day(now.astimezone()),
            sector_strength_by_etf={k: round(v, 4) for k, v in sector_strength.items()},
            leaders=leaders,
            laggards=laggards,
            cohort_tape_quality=round(cohort_tape_quality, 4),
            session_archetype=session_archetype,
            errors=errors[-10:],
        )
