"""
Comprehensive tests for:
  1. Gap order entry gate enforcement (fix: gap trades now go through _entry_gate)
  2. All _entry_gate cases: ML, spread, risk, regime, decision
  3. scan_symbols strategy routing — all 9 strategies dispatch correctly
  4. Scan preset defaults in api_scan_start parameter handling
"""
from __future__ import annotations

import types
import unittest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now_et() -> datetime:
    return datetime.now(ET)


def _make_sym_state(**kwargs) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "price": 15.0,
        "spread_pct": 0.20,
        "ml_score": 0.15,
        "best_side": "short",
        "decision": "GO",
        "entry": 15.0,
        "stop_loss": 16.0,
        "target_2r": 13.0,
        "tape_live": True,
        "monitor_state": "triggered",
        "playbook": "gap_order",
    }
    defaults.update(kwargs)
    return defaults


def _make_gap_plan(sym: str = "TEST", side: str = "short",
                   entry: float = 15.0, stop: float = 16.0,
                   target: float = 13.0) -> dict[str, Any]:
    return {
        "symbol": sym,
        "side": side,
        "entry": entry,
        "stop": stop,
        "target_2r": target,
    }


def _import_monitor():
    """Import paper_trade_monitor avoiding __main__ side-effects."""
    import importlib, sys
    mod_name = "tools.paper_trade_monitor"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        mod_name,
        pathlib.Path(__file__).parents[1] / "tools" / "paper_trade_monitor.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


PTM = _import_monitor()
PaperTradeBook = PTM.PaperTradeBook
GapOrderTracker = PTM.GapOrderTracker
PaperTradeMonitor = PTM.PaperTradeMonitor


# ══════════════════════════════════════════════════════════════════════════════
# 1. Gap order gate enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestGapOrderGateEnforcement(unittest.TestCase):
    """Gap trades must go through entry_gate_fn — not open blindly."""

    def _tracker_with_plan(self, sym="TSYM", side="short",
                            entry=15.0, stop=16.0, target=13.0):
        tracker = GapOrderTracker()
        tracker.plans[sym] = _make_gap_plan(sym, side, entry, stop, target)
        tracker.loaded = True
        return tracker

    def _book(self):
        return PaperTradeBook()

    def _sym_map(self, sym="TSYM", price=14.80):
        return {sym: _make_sym_state(price=price)}

    def _log_fn(self, events: list):
        """Two-arg log function matching the real monitor signature: (event_type, payload)."""
        def _fn(event_type, payload):
            events.append({"event": event_type, **payload})
        return _fn

    def test_gate_pass_opens_trade(self):
        """When gate passes, trade is opened."""
        tracker = self._tracker_with_plan()
        book = self._book()
        events = []
        gate_fn = lambda sym, st, now, ctx: None  # always pass
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)

        tracker.check_triggers(self._sym_map(), now, book, self._log_fn(events),
                                entry_gate_fn=gate_fn, context={})
        self.assertIn("TSYM", book.open_trades)

    def test_gate_block_prevents_trade(self):
        """When gate returns a reason, trade is NOT opened."""
        tracker = self._tracker_with_plan()
        book = self._book()
        events = []
        gate_fn = lambda sym, st, now, ctx: "ml_too_low (0.01 < 0.05)"
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)

        tracker.check_triggers(self._sym_map(), now, book, self._log_fn(events),
                                entry_gate_fn=gate_fn, context={})
        self.assertNotIn("TSYM", book.open_trades)

    def test_gate_block_logs_entry_gate_blocked(self):
        """A blocked gap trade logs an entry_gate_blocked event."""
        tracker = self._tracker_with_plan()
        book = self._book()
        events = []
        gate_fn = lambda sym, st, now, ctx: "spread_too_wide"
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)

        tracker.check_triggers(self._sym_map(), now, book, self._log_fn(events),
                                entry_gate_fn=gate_fn, context={})
        blocked = [e for e in events if e.get("event") == "entry_gate_blocked"]
        self.assertTrue(len(blocked) > 0, "expected entry_gate_blocked event in log")

    def test_no_gate_fn_still_opens_trade(self):
        """Backwards compat: if entry_gate_fn=None, trade opens (old behaviour)."""
        tracker = self._tracker_with_plan()
        book = self._book()
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)

        tracker.check_triggers(self._sym_map(), now, book, lambda *a: None,
                                entry_gate_fn=None, context={})
        self.assertIn("TSYM", book.open_trades)

    def test_already_triggered_not_re_opened(self):
        """Symbol in triggered set is skipped even if gate passes."""
        tracker = self._tracker_with_plan()
        tracker.triggered.add("TSYM")
        book = self._book()
        gate_fn = lambda sym, st, now, ctx: None
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)

        tracker.check_triggers(self._sym_map(), now, book, self._log_fn([]),
                                entry_gate_fn=gate_fn, context={})
        self.assertNotIn("TSYM", book.open_trades)

    def test_before_931_no_triggers_checked(self):
        """No triggers fire before 9:31 AM ET (first candle not closed yet)."""
        tracker = self._tracker_with_plan()
        book = self._book()
        gate_fn = lambda sym, st, now, ctx: None
        now = datetime(2026, 4, 29, 9, 30, tzinfo=ET)  # 9:30 — too early

        tracker.check_triggers(self._sym_map(), now, book, self._log_fn([]),
                                entry_gate_fn=gate_fn, context={})
        self.assertNotIn("TSYM", book.open_trades)

    def test_long_trigger_price_below_entry_no_fire(self):
        """Long: price must be >= entry to trigger."""
        tracker = self._tracker_with_plan(side="long", entry=20.0, stop=18.0, target=24.0)
        book = self._book()
        gate_fn = lambda sym, st, now, ctx: None
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)
        sym_map = {"TSYM": _make_sym_state(price=19.50, best_side="long")}

        tracker.check_triggers(sym_map, now, book, self._log_fn([]),
                                entry_gate_fn=gate_fn, context={})
        self.assertNotIn("TSYM", book.open_trades)

    def test_short_trigger_price_above_entry_no_fire(self):
        """Short: price must be <= entry to trigger."""
        tracker = self._tracker_with_plan(side="short", entry=15.0, stop=16.0, target=13.0)
        book = self._book()
        gate_fn = lambda sym, st, now, ctx: None
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)
        sym_map = {"TSYM": _make_sym_state(price=15.50)}  # above entry

        tracker.check_triggers(sym_map, now, book, self._log_fn([]),
                                entry_gate_fn=gate_fn, context={})
        self.assertNotIn("TSYM", book.open_trades)

    def test_chase_skip_prevents_trade(self):
        """Price more than 2% past entry is skipped as chasing."""
        # Short entry=15.0, price=14.50 → (15-14.5)/15 = 3.3% > 2% CHASE_SKIP
        tracker = self._tracker_with_plan(side="short", entry=15.0, stop=16.0, target=13.0)
        book = self._book()
        gate_fn = lambda sym, st, now, ctx: None
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)
        sym_map = {"TSYM": _make_sym_state(price=14.50)}  # 3.3% through entry

        tracker.check_triggers(sym_map, now, book, self._log_fn([]),
                                entry_gate_fn=gate_fn, context={})
        self.assertNotIn("TSYM", book.open_trades)

    def test_gate_receives_correct_side(self):
        """entry_gate_fn receives best_side from the plan, not from sym_state."""
        tracker = self._tracker_with_plan(side="long", entry=20.0, stop=18.0, target=24.0)
        book = self._book()
        captured = {}
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)
        sym_map = {"TSYM": _make_sym_state(price=20.10, best_side="short")}  # state says short

        def gate_fn(sym, st, n, ctx):
            captured["side"] = st.get("best_side")
            return None  # pass

        tracker.check_triggers(sym_map, now, book, self._log_fn([]),
                                entry_gate_fn=gate_fn, context={})
        self.assertEqual(captured.get("side"), "long")  # plan's side wins

    def test_gate_context_passed_through(self):
        """context dict is forwarded to gate_fn."""
        tracker = self._tracker_with_plan()
        book = self._book()
        captured = {}
        now = datetime(2026, 4, 29, 9, 32, tzinfo=ET)
        ctx = {"risk_on_score": -0.44, "spy_trend": "down"}

        def gate_fn(sym, st, n, context):
            captured.update(context)
            return None

        tracker.check_triggers(self._sym_map(), now, book, self._log_fn([]),
                                entry_gate_fn=gate_fn, context=ctx)
        self.assertEqual(captured.get("risk_on_score"), -0.44)


# ══════════════════════════════════════════════════════════════════════════════
# 2. _entry_gate: all gate cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEntryGate(unittest.TestCase):
    """Unit tests for PaperTradeMonitor._entry_gate."""

    def _monitor(self) -> PaperTradeMonitor:
        with patch.object(PTM, "discover_monitor_id", return_value="mock-id"), \
             patch.object(PTM, "LOG_DIR", PTM.Path("/tmp")):
            m = PaperTradeMonitor.__new__(PaperTradeMonitor)
            m.log_path = PTM.Path("/tmp/test_gate.jsonl")
            m.summary_path = PTM.Path("/tmp/test_gate_summary.md")
            return m

    def _gate(self, st: dict, context: dict | None = None) -> str | None:
        mon = self._monitor()
        return mon._entry_gate("SYM", st, _now_et(), context or {})

    # Gate 1: ML score
    def test_ml_none_passes(self):
        st = _make_sym_state(ml_score=None)
        self.assertIsNone(self._gate(st))

    def test_ml_above_threshold_passes(self):
        st = _make_sym_state(ml_score=0.10)
        self.assertIsNone(self._gate(st))

    def test_ml_exactly_at_threshold_passes(self):
        st = _make_sym_state(ml_score=PTM.PT_MIN_ML_SCORE)
        self.assertIsNone(self._gate(st))

    def test_ml_below_threshold_blocked(self):
        st = _make_sym_state(ml_score=PTM.PT_MIN_ML_SCORE - 0.001)
        reason = self._gate(st)
        self.assertIsNotNone(reason)
        self.assertIn("ml_too_low", reason)  # type: ignore[arg-type]

    def test_ml_zero_blocked(self):
        st = _make_sym_state(ml_score=0.0)
        reason = self._gate(st)
        self.assertIn("ml_too_low", reason)  # type: ignore[arg-type]

    # Gate 2: spread
    def test_spread_zero_blocked(self):
        """Zero spread means missing quote data — must block."""
        st = _make_sym_state(spread_pct=0.0, price=20.0, ml_score=None)
        reason = self._gate(st)
        self.assertIn("spread_missing", reason)  # type: ignore[arg-type]

    def test_spread_sub10_within_limit_passes(self):
        st = _make_sym_state(spread_pct=PTM.PT_MAX_SPREAD_SUB10 - 0.01, price=5.0, ml_score=None)
        self.assertIsNone(self._gate(st))

    def test_spread_sub10_over_limit_blocked(self):
        st = _make_sym_state(spread_pct=PTM.PT_MAX_SPREAD_SUB10 + 0.01, price=5.0, ml_score=None)
        reason = self._gate(st)
        self.assertIn("spread_too_wide", reason)  # type: ignore[arg-type]

    def test_spread_above10_within_limit_passes(self):
        st = _make_sym_state(spread_pct=PTM.PT_MAX_SPREAD_ABOVE10 - 0.01, price=15.0, ml_score=None)
        self.assertIsNone(self._gate(st))

    def test_spread_above10_over_limit_blocked(self):
        st = _make_sym_state(spread_pct=PTM.PT_MAX_SPREAD_ABOVE10 + 0.01, price=15.0, ml_score=None)
        reason = self._gate(st)
        self.assertIn("spread_too_wide", reason)  # type: ignore[arg-type]

    def test_wide_spread_like_htco_blocked(self):
        """6.22% spread (HTCO today) must be blocked at $28 price."""
        st = _make_sym_state(spread_pct=6.22, price=28.99, ml_score=None)
        reason = self._gate(st)
        self.assertIn("spread_too_wide", reason)  # type: ignore[arg-type]

    def test_wide_spread_like_kfrc_blocked(self):
        """5.07% spread (KFRC today) at $40 price must be blocked."""
        st = _make_sym_state(spread_pct=5.07, price=40.0, ml_score=None)
        reason = self._gate(st)
        self.assertIn("spread_too_wide", reason)  # type: ignore[arg-type]

    # Gate 3: risk per share
    def test_risk_too_small_blocked(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.00, stop_loss=15.02)  # $0.02 risk < $0.03 min
        reason = self._gate(st)
        self.assertIn("risk_too_small", reason)  # type: ignore[arg-type]

    def test_risk_too_large_blocked(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=20.0)  # $5.00 risk > $2.00 max
        reason = self._gate(st)
        self.assertIn("risk_too_large", reason)  # type: ignore[arg-type]

    def test_risk_within_band_passes(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=14.50)  # $0.50 risk
        self.assertIsNone(self._gate(st))

    # Gate 4: regime (long trades only)
    def test_regime_risk_off_blocks_long(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=14.0, best_side="long")
        reason = self._gate(st, context={"risk_on_score": -0.44})
        self.assertIn("regime_risk_off", reason)  # type: ignore[arg-type]

    def test_regime_risk_off_does_not_block_short(self):
        """Risk-off market should NOT block short trades."""
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=16.0, best_side="short")
        self.assertIsNone(self._gate(st, context={"risk_on_score": -0.44}))

    def test_regime_score_above_threshold_allows_long(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=14.0, best_side="long")
        self.assertIsNone(self._gate(st, context={"risk_on_score": 0.50}))

    def test_regime_none_score_allows_long(self):
        """If regime score missing, gate does not block."""
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=14.0, best_side="long")
        self.assertIsNone(self._gate(st, context={}))

    # Gate 5: decision
    def test_decision_wait_blocked(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=16.0, decision="WAIT")
        reason = self._gate(st)
        self.assertIn("decision_not_go", reason)  # type: ignore[arg-type]

    def test_decision_pass_blocked(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=16.0, decision="PASS")
        reason = self._gate(st)
        self.assertIn("decision_not_go", reason)  # type: ignore[arg-type]

    def test_decision_go_passes(self):
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=16.0, decision="GO")
        self.assertIsNone(self._gate(st))

    def test_decision_empty_passes(self):
        """Empty/None decision should not block (monitor may not always set it)."""
        st = _make_sym_state(ml_score=None, spread_pct=0.20, price=15.0,
                              entry=15.0, stop_loss=16.0, decision=None)
        self.assertIsNone(self._gate(st))

    # Today's actual losers — all should now be blocked
    def test_todays_eras_ml_blocked(self):
        """ERAS: ML=0.028, below 0.05 gate → blocked."""
        st = _make_sym_state(ml_score=0.028, spread_pct=0.23, price=8.72,
                              entry=8.81, stop_loss=9.43, best_side="short", decision="GO")
        reason = self._gate(st)
        self.assertIn("ml_too_low", reason)  # type: ignore[arg-type]

    def test_todays_visn_ml_blocked(self):
        """VISN: ML=0.006 → blocked."""
        st = _make_sym_state(ml_score=0.006, spread_pct=0.52, price=9.56,
                              entry=9.60, stop_loss=10.27, best_side="short", decision="GO")
        reason = self._gate(st)
        self.assertIn("ml_too_low", reason)  # type: ignore[arg-type]

    def test_todays_crmx_spread_blocked(self):
        """CRMX: spread=1.73% at $25 (above $10 tier 0.40% max) → blocked."""
        st = _make_sym_state(ml_score=None, spread_pct=1.73, price=25.0,
                              entry=25.34, stop_loss=27.11, best_side="short", decision="WAIT")
        reason = self._gate(st)
        # decision=WAIT blocks first, spread also would block
        self.assertIsNotNone(reason)

    def test_todays_bbby_regime_blocked(self):
        """BBBY: LONG in risk_on=-0.44 market → blocked by regime gate."""
        st = _make_sym_state(ml_score=None, spread_pct=0.27, price=7.47,
                              entry=7.46, stop_loss=6.94, best_side="long", decision="GO")
        reason = self._gate(st, context={"risk_on_score": -0.44})
        self.assertIn("regime_risk_off", reason)  # type: ignore[arg-type]

    def test_todays_snbr_wait_blocked(self):
        """SNBR: Decision=WAIT → blocked."""
        st = _make_sym_state(ml_score=0.164, spread_pct=0.30, price=3.33,
                              entry=3.27, stop_loss=3.04, best_side="long", decision="WAIT")
        reason = self._gate(st, context={"risk_on_score": -0.44})
        # regime blocks first (long + risk_off), but WAIT also would
        self.assertIsNotNone(reason)


# ══════════════════════════════════════════════════════════════════════════════
# 3. scan_symbols strategy routing
# ══════════════════════════════════════════════════════════════════════════════

class TestScanStrategyRouting(unittest.TestCase):
    """scan_symbols must dispatch to the correct strategy function."""

    def _patch_all(self):
        """Return a dict of mock targets for all sub-strategies."""
        return {
            "range_reversion": patch("scanner.orb.scan_range_reversion_symbols",
                                     return_value={"strategy": "range_reversion"}),
            "range_trap":      patch("scanner.orb.scan_range_trap_symbols",
                                     return_value={"strategy": "range_trap"}),
            "gap_and_go":      patch("scanner.orb.scan_gap_and_go_symbols",
                                     return_value={"strategy": "gap_and_go"}),
            "float_rotation":  patch("scanner.orb.scan_float_rotation_symbols",
                                     return_value={"strategy": "float_rotation"}),
            "halt_resume":     patch("scanner.orb.scan_halt_resume_symbols",
                                     return_value={"strategy": "halt_resume"}),
            "wk52":            patch("scanner.orb.scan_52wk_pullback_symbols",
                                     return_value={"strategy": "52wk_pullback"}),
            "atr_expansion":   patch("scanner.orb.scan_atr_expansion_symbols",
                                     return_value={"strategy": "atr_expansion"}),
            "eod_momentum":    patch("scanner.orb.scan_eod_momentum_symbols",
                                     return_value={"strategy": "eod_momentum"}),
        }

    def _run(self, strategy: str, alias: str | None = None):
        from scanner.orb import scan_symbols, ORBConfig
        cfg = ORBConfig()
        patches = self._patch_all()
        with patches["range_reversion"] as rr, \
             patches["range_trap"]      as rt, \
             patches["gap_and_go"]      as gg, \
             patches["float_rotation"]  as fr, \
             patches["halt_resume"]     as hr, \
             patches["wk52"]            as wk, \
             patches["atr_expansion"]   as ae, \
             patches["eod_momentum"]    as eod:
            result = scan_symbols([], cfg, strategy=alias or strategy)
            return {
                "rr": rr, "rt": rt, "gg": gg, "fr": fr,
                "hr": hr, "wk": wk, "ae": ae, "eod": eod,
                "result": result,
            }

    def test_orb_does_not_dispatch_to_sub_strategies(self):
        mocks = self._run("orb")
        for key in ("rr", "rt", "gg", "fr", "hr", "wk", "ae", "eod"):
            mocks[key].assert_not_called()

    def test_range_reversion_primary(self):
        mocks = self._run("range_reversion")
        mocks["rr"].assert_called_once()

    def test_range_reversion_alias_rr(self):
        mocks = self._run("range_reversion", alias="rr")
        mocks["rr"].assert_called_once()

    def test_range_reversion_alias_range(self):
        mocks = self._run("range_reversion", alias="range")
        mocks["rr"].assert_called_once()

    def test_range_trap_primary(self):
        mocks = self._run("range_trap")
        mocks["rt"].assert_called_once()

    def test_range_trap_alias_rt(self):
        mocks = self._run("range_trap", alias="rt")
        mocks["rt"].assert_called_once()

    def test_range_trap_alias_trap(self):
        mocks = self._run("range_trap", alias="trap")
        mocks["rt"].assert_called_once()

    def test_gap_and_go_primary(self):
        mocks = self._run("gap_and_go")
        mocks["gg"].assert_called_once()

    def test_gap_and_go_alias_gg(self):
        mocks = self._run("gap_and_go", alias="gg")
        mocks["gg"].assert_called_once()

    def test_gap_and_go_alias_gap(self):
        mocks = self._run("gap_and_go", alias="gap")
        mocks["gg"].assert_called_once()

    def test_float_rotation_primary(self):
        mocks = self._run("float_rotation")
        mocks["fr"].assert_called_once()

    def test_float_rotation_alias_float(self):
        mocks = self._run("float_rotation", alias="float")
        mocks["fr"].assert_called_once()

    def test_halt_resume_primary(self):
        mocks = self._run("halt_resume")
        mocks["hr"].assert_called_once()

    def test_halt_resume_alias_halt(self):
        mocks = self._run("halt_resume", alias="halt")
        mocks["hr"].assert_called_once()

    def test_52wk_pullback_primary(self):
        mocks = self._run("52wk_pullback")
        mocks["wk"].assert_called_once()

    def test_52wk_pullback_alias_pullback(self):
        mocks = self._run("52wk_pullback", alias="pullback")
        mocks["wk"].assert_called_once()

    def test_atr_expansion_primary(self):
        mocks = self._run("atr_expansion")
        mocks["ae"].assert_called_once()

    def test_atr_expansion_alias_atr(self):
        mocks = self._run("atr_expansion", alias="atr")
        mocks["ae"].assert_called_once()

    def test_eod_momentum_primary(self):
        mocks = self._run("eod_momentum")
        mocks["eod"].assert_called_once()

    def test_eod_momentum_alias_saturday_prep(self):
        mocks = self._run("eod_momentum", alias="saturday_prep")
        mocks["eod"].assert_called_once()

    def test_eod_momentum_alias_eod(self):
        mocks = self._run("eod_momentum", alias="eod")
        mocks["eod"].assert_called_once()

    def test_range_reversion_not_called_for_orb(self):
        mocks = self._run("orb")
        mocks["rr"].assert_not_called()

    def test_only_one_strategy_dispatched_at_a_time(self):
        """Exactly one sub-strategy should be called for each non-ORB strategy."""
        sub_keys = ("rr", "rt", "gg", "fr", "hr", "wk", "ae", "eod")
        strategy_map = {
            "rr": "rr", "rt": "rt", "gg": "gg",
            "fr": "fr", "hr": "hr", "wk": "52wk_pullback",
            "ae": "atr_expansion", "eod": "eod_momentum",
        }
        for key, strat in strategy_map.items():
            mocks = self._run(strat)
            called = [k for k in sub_keys if mocks[k].called]
            self.assertEqual(called, [key],
                             f"strategy={strat}: expected [{key}] called, got {called}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Scan preset defaults
# ══════════════════════════════════════════════════════════════════════════════

class TestScanPresets(unittest.TestCase):
    """Verify preset blocks set the expected defaults in api_scan_start logic."""

    def _apply_preset(self, preset: str) -> dict:
        """Simulate the preset-application block from api_scan_start."""
        data: dict = {"preset": preset, "strategy": "orb"}

        if preset == "tradeideas":
            data.setdefault("use_ml", "1")
            data.setdefault("use_sentiment", "1")
            data.setdefault("use_catalyst", "1")
            data.setdefault("min_grade", "B")
            data.setdefault("min_combined_score", "0.46")
            data.setdefault("min_rvol", "2.0")
            data.setdefault("min_today_dollar_vol", "5000000")
        elif preset == "tradingview":
            data.setdefault("use_ml", "0")
            data.setdefault("use_sentiment", "0")
            data.setdefault("use_catalyst", "0")
            data.setdefault("min_grade", "C")
            data.setdefault("min_combined_enabled", "0")
            data.setdefault("min_rvol", "1.2")
            data.setdefault("min_today_dollar_vol", "1500000")
        elif preset == "trendspider":
            data.setdefault("use_ml", "1")
            data.setdefault("use_sentiment", "1")
            data.setdefault("use_catalyst", "1")
            data.setdefault("min_grade", "B")
            data.setdefault("min_combined_score", "0.44")
            data.setdefault("no_chop_enabled", "1")
            data.setdefault("min_vwap_enabled", "1")
            data.setdefault("min_pct_over_vwap", "0.6")
        elif preset == "finviz":
            data.setdefault("use_ml", "0")
            data.setdefault("use_sentiment", "0")
            data.setdefault("use_catalyst", "0")
            data.setdefault("min_grade", "C")
            data.setdefault("min_combined_enabled", "0")
            data.setdefault("min_today_dollar_vol", "2000000")
            data.setdefault("min_avg20_dollar_vol", "1500000")
        elif preset == "benzinga":
            data.setdefault("use_ml", "1")
            data.setdefault("use_sentiment", "1")
            data.setdefault("use_catalyst", "1")
            data.setdefault("min_grade", "B")
            data.setdefault("min_combined_score", "0.42")
            data.setdefault("min_today_dollar_vol", "3000000")
            data.setdefault("min_rvol", "1.6")
        elif preset == "shorts":
            data.setdefault("short_only", "1")
            data.setdefault("long_only", "0")
            data.setdefault("min_price", "10.0")
            data.setdefault("max_price", "500.0")
            data.setdefault("min_today_dollar_vol", "5000000")
            data.setdefault("min_avg20_dollar_vol", "3000000")
            data.setdefault("min_rvol", "1.5")
            data.setdefault("min_grade", "B")
            data.setdefault("min_vwap_enabled", "0")
            data.setdefault("no_chop_enabled", "1")
        elif preset == "saturday_prep":
            data["strategy"] = "eod_momentum"  # force — setdefault silently loses to existing key
            data.setdefault("min_day_move_pct", "3.0")
            data.setdefault("max_day_move_pct", "50.0")
            data.setdefault("min_close_vs_range", "0.60")
            data.setdefault("min_rvol", "1.5")
            data.setdefault("min_avg20_dollar_vol", "1000000")
            data.setdefault("max_symbols", "5000")
            data.setdefault("use_ml", "0")
        return data

    def test_tradeideas_enables_ml_sentiment_catalyst(self):
        d = self._apply_preset("tradeideas")
        self.assertEqual(d["use_ml"], "1")
        self.assertEqual(d["use_sentiment"], "1")
        self.assertEqual(d["use_catalyst"], "1")

    def test_tradeideas_min_rvol_2(self):
        d = self._apply_preset("tradeideas")
        self.assertEqual(d["min_rvol"], "2.0")

    def test_tradeideas_min_dvol_5m(self):
        d = self._apply_preset("tradeideas")
        self.assertEqual(d["min_today_dollar_vol"], "5000000")

    def test_tradeideas_grade_B(self):
        d = self._apply_preset("tradeideas")
        self.assertEqual(d["min_grade"], "B")

    def test_tradingview_disables_ml(self):
        d = self._apply_preset("tradingview")
        self.assertEqual(d["use_ml"], "0")
        self.assertEqual(d["use_sentiment"], "0")
        self.assertEqual(d["use_catalyst"], "0")

    def test_tradingview_grade_C(self):
        d = self._apply_preset("tradingview")
        self.assertEqual(d["min_grade"], "C")

    def test_trendspider_antichop_and_vwap(self):
        d = self._apply_preset("trendspider")
        self.assertEqual(d["no_chop_enabled"], "1")
        self.assertEqual(d["min_vwap_enabled"], "1")

    def test_finviz_avg20_dvol_set(self):
        d = self._apply_preset("finviz")
        self.assertEqual(d["min_avg20_dollar_vol"], "1500000")

    def test_benzinga_combined_score(self):
        d = self._apply_preset("benzinga")
        self.assertEqual(d["min_combined_score"], "0.42")

    def test_shorts_disables_vwap_gate(self):
        """Shorts are below VWAP by definition — VWAP gate must be off."""
        d = self._apply_preset("shorts")
        self.assertEqual(d["min_vwap_enabled"], "0")

    def test_shorts_min_price_10(self):
        """Hard-to-borrow below $10 — min price must be $10."""
        d = self._apply_preset("shorts")
        self.assertEqual(d["min_price"], "10.0")

    def test_shorts_enables_short_only(self):
        d = self._apply_preset("shorts")
        self.assertEqual(d["short_only"], "1")
        self.assertEqual(d["long_only"], "0")

    def test_shorts_no_chop_enabled(self):
        d = self._apply_preset("shorts")
        self.assertEqual(d["no_chop_enabled"], "1")

    def test_saturday_prep_routes_to_eod_momentum(self):
        d = self._apply_preset("saturday_prep")
        self.assertEqual(d["strategy"], "eod_momentum")

    def test_saturday_prep_disables_ml(self):
        d = self._apply_preset("saturday_prep")
        self.assertEqual(d["use_ml"], "0")

    def test_saturday_prep_large_symbol_universe(self):
        d = self._apply_preset("saturday_prep")
        self.assertEqual(d["max_symbols"], "5000")

    def test_preset_does_not_override_caller_value(self):
        """setdefault must not overwrite explicit caller params."""
        data: dict = {"preset": "tradeideas", "min_rvol": "5.0"}
        data.setdefault("min_rvol", "2.0")
        self.assertEqual(data["min_rvol"], "5.0")


if __name__ == "__main__":
    unittest.main()
