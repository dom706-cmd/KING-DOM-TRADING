"""
Correctness regression tests for the three fixes from the 2026-03-31 postmortem:
  1. Spread threshold is now price-tiered (sub-$5: 2.0%, else 1.5%) — not flat 0.75%
  2. Near-zero ml_score (diagnostic-only) does NOT produce a PASS decision (should be WAIT)
  3. to_api() rounds price/entry/stop/target fields to 4 decimal places
"""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from monitor import live_monitor
from monitor.live_monitor import (
    LiveMonitorManager,
    MonitorSession,
    MonitorSymbolState,
)


def _make_session() -> MonitorSession:
    return MonitorSession(
        monitor_id="test-session",
        job_id=None,
        feed_requested="sip",
        feed_used="sip",
        source="test",
    )


def _make_st(symbol: str = "TSYM", price: float = 1.50) -> MonitorSymbolState:
    st = MonitorSymbolState(symbol=symbol)
    st.price = price
    st.bid = price - 0.01
    st.ask = price + 0.01
    st.market_session_state = "regular"
    # set ages to fresh
    st.quote_age_ms = 100
    st.trade_age_ms = 100
    # compute spread
    mid = (st.bid + st.ask) / 2.0
    st.spread_pct = ((st.ask - st.bid) / mid) * 100.0
    return st


class SpreadThresholdTests(unittest.TestCase):
    """Fix 1: Spread threshold is price-tiered — not a flat 0.75%."""

    def test_sub5_with_2pct_spread_is_tape_live(self) -> None:
        """$1.50 stock with exactly 2.0% spread should be accepted (new limit = 2.0%)."""
        sess = _make_session()
        st = MonitorSymbolState(symbol="CHEAP")
        st.price = 1.50
        st.quote_age_ms = 100
        st.trade_age_ms = 100
        st.market_session_state = "regular"
        # 2.0% spread: bid=1.485, ask=1.515 → spread = 0.03 / 1.50 = 2.0%
        st.bid = 1.485
        st.ask = 1.515
        mid = (st.bid + st.ask) / 2.0
        st.spread_pct = round((st.ask - st.bid) / mid * 100.0, 6)
        now_ts = time.time()
        LiveMonitorManager._compute_tape_locked(None, sess, st, now_ts=now_ts)  # type: ignore[arg-type]
        self.assertTrue(st.tape_live, f"tape rejected for 2.0% spread on sub-$5 name: {st.tape_live_reason}")

    def test_sub5_exceeds_2pct_spread_is_rejected(self) -> None:
        """$1.50 stock with 2.5% spread should be rejected."""
        sess = _make_session()
        st = MonitorSymbolState(symbol="CHEAP")
        st.price = 1.50
        st.quote_age_ms = 100
        st.trade_age_ms = 100
        st.market_session_state = "regular"
        # 2.5% spread
        st.bid = 1.48125
        st.ask = 1.51875
        mid = (st.bid + st.ask) / 2.0
        st.spread_pct = round((st.ask - st.bid) / mid * 100.0, 6)
        now_ts = time.time()
        LiveMonitorManager._compute_tape_locked(None, sess, st, now_ts=now_ts)  # type: ignore[arg-type]
        self.assertFalse(st.tape_live)
        self.assertEqual(st.tape_live_reason, "wide_spread")

    def test_above5_1pt5pct_spread_is_tape_live(self) -> None:
        """$6.00 stock with exactly 1.5% spread passes (new limit = 1.5%)."""
        sess = _make_session()
        st = MonitorSymbolState(symbol="MID")
        st.price = 6.00
        st.quote_age_ms = 100
        st.trade_age_ms = 100
        st.market_session_state = "regular"
        # 1.5% spread on $6 = $0.09
        st.bid = 5.955
        st.ask = 6.045
        mid = (st.bid + st.ask) / 2.0
        st.spread_pct = round((st.ask - st.bid) / mid * 100.0, 6)
        now_ts = time.time()
        LiveMonitorManager._compute_tape_locked(None, sess, st, now_ts=now_ts)  # type: ignore[arg-type]
        self.assertTrue(st.tape_live, f"expected tape_live, got: {st.tape_live_reason}")

    def test_above5_exceeds_1pt5pct_spread_is_rejected(self) -> None:
        """$6.00 stock with 1.8% spread is rejected."""
        sess = _make_session()
        st = MonitorSymbolState(symbol="MID")
        st.price = 6.00
        st.quote_age_ms = 100
        st.trade_age_ms = 100
        st.market_session_state = "regular"
        # 1.8% spread
        st.bid = 5.946
        st.ask = 6.054
        mid = (st.bid + st.ask) / 2.0
        st.spread_pct = round((st.ask - st.bid) / mid * 100.0, 6)
        now_ts = time.time()
        LiveMonitorManager._compute_tape_locked(None, sess, st, now_ts=now_ts)  # type: ignore[arg-type]
        self.assertFalse(st.tape_live)
        self.assertEqual(st.tape_live_reason, "wide_spread")


class P2r30mFallbackTests(unittest.TestCase):
    """Fix 2: Near-zero ml_score as p_2r_30m proxy must yield WAIT not PASS."""

    def _seed_state(self, candidate: dict) -> MonitorSymbolState:
        mgr = LiveMonitorManager.__new__(LiveMonitorManager)
        return mgr._make_seed_state(candidate, idx=1, playbook="open_drive_orb", seed_source="scan")

    def test_zero_ml_score_yields_wait_not_pass(self) -> None:
        """ml_score=0.0 (diagnostic-only) must not become p_2r_30m=0.0 → PASS."""
        c = {
            "symbol": "DIAG",
            "best_side": "long",
            "entry": 3.56,
            "stop_loss": 3.20,
            "target_2r": 4.28,
            "target_3r": 4.64,
            "risk_per_share": 0.36,
            "ml_score": 0.0,
        }
        st = self._seed_state(c)
        # p_2r_30m must be None (not 0.0) so build_plan_state returns WAIT not PASS
        self.assertIsNone(st.p_2r_30m, f"expected None, got {st.p_2r_30m}")
        self.assertIsNone(st.probability_source)

    def test_low_ml_score_below_threshold_yields_none(self) -> None:
        """ml_score=0.05 is below the 0.10 floor — must not pollute p_2r_30m."""
        c = {
            "symbol": "LOW",
            "best_side": "long",
            "entry": 2.50,
            "stop_loss": 2.20,
            "target_2r": 3.10,
            "target_3r": 3.40,
            "risk_per_share": 0.30,
            "ml_score": 0.05,
        }
        st = self._seed_state(c)
        self.assertIsNone(st.p_2r_30m)

    def test_meaningful_ml_score_propagates_as_p2r_30m(self) -> None:
        """ml_score=0.65 should propagate to p_2r_30m for plan state evaluation."""
        c = {
            "symbol": "GOOD",
            "best_side": "long",
            "entry": 5.00,
            "stop_loss": 4.50,
            "target_2r": 6.00,
            "target_3r": 6.50,
            "risk_per_share": 0.50,
            "ml_score": 0.65,
        }
        st = self._seed_state(c)
        self.assertAlmostEqual(st.p_2r_30m or 0.0, 0.65, places=4)
        self.assertEqual(st.probability_source, "scan_ml_score")

    def test_explicit_p_2r_30m_always_wins(self) -> None:
        """Explicit p_2r_30m=0.32 must be used even when ml_score is also present."""
        c = {
            "symbol": "BOTH",
            "best_side": "long",
            "entry": 4.00,
            "stop_loss": 3.60,
            "target_2r": 4.80,
            "target_3r": 5.20,
            "risk_per_share": 0.40,
            "p_2r_30m": 0.32,
            "ml_score": 0.90,
        }
        st = self._seed_state(c)
        self.assertAlmostEqual(st.p_2r_30m or 0.0, 0.32, places=4)
        self.assertEqual(st.probability_source, "entry_now")


class ToApiPriceRoundingTests(unittest.TestCase):
    """Fix 3: to_api() must round price/entry/stop/target to 4 decimal places."""

    def test_float_imprecision_is_rounded_in_api(self) -> None:
        """3.5599999999999996 must serialize as 3.56 (rounded to 4dp)."""
        st = MonitorSymbolState(symbol="TMQ")
        st.entry = 3.5599999999999996  # real float artifact from OR computation
        st.stop_loss = 3.2689000000000001
        st.target_2r = 3.8509000000000002
        st.target_3r = 4.1419000000000003
        st.risk_per_share = 0.29100000000000002
        st.or_high = 3.56
        st.or_low = 3.27
        st.vwap_last = 3.4499999999999997
        st.scan_price = 3.5599999999999996

        api = st.to_api()

        self.assertEqual(api["entry"], 3.56)
        self.assertEqual(api["stop_loss"], 3.2689)
        self.assertEqual(api["risk_per_share"], 0.291)
        self.assertEqual(api["vwap_last"], 3.45)
        self.assertEqual(api["scan_price"], 3.56)

        ep = api["execution_plan"]
        self.assertEqual(ep["entry"], 3.56)
        self.assertEqual(ep["stop_loss"], 3.2689)

    def test_none_price_fields_stay_none(self) -> None:
        """None price fields must not be converted to 0.0 by rounding."""
        st = MonitorSymbolState(symbol="NONE")
        api = st.to_api()
        self.assertIsNone(api["entry"])
        self.assertIsNone(api["stop_loss"])
        self.assertIsNone(api["target_2r"])
        self.assertIsNone(api["execution_plan"]["entry"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
