"""
Test: OR and VWAP are filled for watchlist-seeded monitor symbols.

Root cause: _build_seed_from_watchlist_plan() always sets or_high/or_low/vwap_last=None.
Fix: _maybe_upgrade_snapshot_seeds() now has a second block that fills these for
seed_source=="watchlist" symbols once 9:31+ ET and bars are available.
"""
from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from monitor.live_monitor import LiveMonitorManager, MonitorSession, MonitorSymbolState


def _make_manager() -> LiveMonitorManager:
    mgr = LiveMonitorManager.__new__(LiveMonitorManager)
    import threading
    mgr._lock = threading.Lock()
    mgr._sessions: dict = {}
    mgr._provider = None
    mgr._stream_cache = None
    mgr._store = None
    mgr._context_engine = None
    mgr._pubsub = None
    return mgr


def _make_session(monitor_id: str = "test-001") -> MonitorSession:
    return MonitorSession(
        monitor_id=monitor_id,
        job_id=None,
        feed_requested="sip",
        feed_used="sip",
        source="test",
    )


def _make_watchlist_state(symbol: str = "AAPL") -> MonitorSymbolState:
    """Simulate a symbol added from the desk watchlist — no OR/VWAP yet."""
    st = MonitorSymbolState(symbol=symbol)
    st.seed_source = "watchlist"
    st.or_high = None
    st.or_low = None
    st.vwap_last = None
    st.entry = 195.0
    st.stop_loss = 192.0
    st.target_2r = 201.0
    return st


def _fake_bars(n: int = 10) -> pd.DataFrame:
    """Synthetic 1-minute OHLCV bars: first 5 bars define the OR."""
    idx = pd.date_range("2026-04-21 09:30", periods=n, freq="1min", tz="America/New_York")
    data = {
        "open":   [100.0] * n,
        "high":   [101.0, 102.0, 103.0, 104.0, 105.0] + [106.0] * (n - 5),
        "low":    [99.0,  98.0,  97.0,  96.0,  95.0]  + [94.0]  * (n - 5),
        "close":  [100.5] * n,
        "volume": [10000] * n,
    }
    return pd.DataFrame(data, index=idx)


class WatchlistORVwapTests(unittest.TestCase):

    def _run_upgrade(self, mgr: LiveMonitorManager, monitor_id: str, provider, wait_s: float = 2.0) -> None:
        """Call _maybe_upgrade_snapshot_seeds at 10:00 AM ET and wait for background thread."""
        et = datetime(2026, 4, 21, 14, 0, 0, tzinfo=timezone.utc)  # 10:00 AM ET
        now_ts = et.timestamp()
        mgr._maybe_upgrade_snapshot_seeds(monitor_id, provider=provider, now_ts=now_ts)
        # OR/VWAP fill runs in a background daemon thread — wait for it to complete
        deadline = time.time() + wait_s
        while time.time() < deadline:
            sess = mgr._sessions.get(monitor_id)
            if sess is None:
                break
            # All symbols have left the sentinel (-1.0) state → thread done
            if all(st.or_high != -1.0 for st in sess.symbols.values()):
                break
            time.sleep(0.05)

    def test_or_high_low_populated(self) -> None:
        """or_high and or_low must be set from the first 5 bars after the fix runs."""
        mgr = _make_manager()
        sess = _make_session()
        st = _make_watchlist_state("TSLA")
        sess.symbols["TSLA"] = st
        mgr._sessions["test-001"] = sess

        bars = _fake_bars(10)
        provider = MagicMock()
        provider.get_bars_range.return_value = bars
        mgr._provider = provider

        with patch("scanner.indicators.vwap", return_value=pd.Series([150.0] * 10)):
            self._run_upgrade(mgr, "test-001", provider)

        self.assertIsNotNone(st.or_high, "or_high should be filled")
        self.assertIsNotNone(st.or_low,  "or_low should be filled")
        # First 5 bars: highs=[101,102,103,104,105], lows=[99,98,97,96,95]
        self.assertAlmostEqual(st.or_high, 105.0)
        self.assertAlmostEqual(st.or_low,   95.0)

    def test_vwap_populated(self) -> None:
        """vwap_last must be set from the indicator."""
        mgr = _make_manager()
        sess = _make_session()
        st = _make_watchlist_state("NVDA")
        sess.symbols["NVDA"] = st
        mgr._sessions["test-001"] = sess

        bars = _fake_bars(10)
        provider = MagicMock()
        provider.get_bars_range.return_value = bars
        mgr._provider = provider

        expected_vwap = 152.34
        with patch("scanner.indicators.vwap", return_value=pd.Series([expected_vwap] * 10)):
            self._run_upgrade(mgr, "test-001", provider)

        self.assertIsNotNone(st.vwap_last, "vwap_last should be filled")
        self.assertAlmostEqual(st.vwap_last, expected_vwap, places=2)

    def test_trader_levels_preserved(self) -> None:
        """entry/stop_loss/target must NOT be overwritten by the OR fill."""
        mgr = _make_manager()
        sess = _make_session()
        st = _make_watchlist_state("AMD")
        original_entry  = st.entry
        original_stop   = st.stop_loss
        original_target = st.target_2r
        sess.symbols["AMD"] = st
        mgr._sessions["test-001"] = sess

        bars = _fake_bars(10)
        provider = MagicMock()
        provider.get_bars_range.return_value = bars
        mgr._provider = provider

        with patch("scanner.indicators.vwap", return_value=pd.Series([150.0] * 10)):
            self._run_upgrade(mgr, "test-001", provider)

        self.assertEqual(st.entry,     original_entry,  "entry must not change")
        self.assertEqual(st.stop_loss, original_stop,   "stop_loss must not change")
        self.assertEqual(st.target_2r, original_target, "target_2r must not change")

    def test_skipped_when_already_filled(self) -> None:
        """Symbols that already have or_high should not trigger another bar fetch."""
        mgr = _make_manager()
        sess = _make_session()
        st = _make_watchlist_state("META")
        st.or_high = 200.0  # already filled
        st.or_low  = 195.0
        sess.symbols["META"] = st
        mgr._sessions["test-001"] = sess

        provider = MagicMock()
        provider.get_bars_range.return_value = _fake_bars(10)
        mgr._provider = provider

        with patch("scanner.indicators.vwap", return_value=pd.Series([150.0] * 10)):
            self._run_upgrade(mgr, "test-001", provider)

        provider.get_bars_range.assert_not_called()

    def test_skipped_before_orb_ready(self) -> None:
        """Before 9:31 ET the entire method should be a no-op."""
        mgr = _make_manager()
        sess = _make_session()
        st = _make_watchlist_state("MSFT")
        sess.symbols["MSFT"] = st
        mgr._sessions["test-001"] = sess

        provider = MagicMock()
        provider.get_bars_range.return_value = _fake_bars(10)
        mgr._provider = provider

        # 9:25 ET — before ORB ready
        et_early = datetime(2026, 4, 21, 13, 25, 0, tzinfo=timezone.utc)  # 9:25 ET (UTC-4)
        with patch("scanner.indicators.vwap", return_value=pd.Series([150.0] * 10)):
            mgr._maybe_upgrade_snapshot_seeds("test-001", provider=provider, now_ts=et_early.timestamp())

        self.assertIsNone(st.or_high, "or_high must remain None before 9:31 ET")
        provider.get_bars_range.assert_not_called()

    def test_no_crash_on_empty_bars(self) -> None:
        """Empty bar response must not raise — symbol stays with or_high=None."""
        mgr = _make_manager()
        sess = _make_session()
        st = _make_watchlist_state("GOOG")
        sess.symbols["GOOG"] = st
        mgr._sessions["test-001"] = sess

        provider = MagicMock()
        provider.get_bars_range.return_value = pd.DataFrame()
        mgr._provider = provider

        with patch("scanner.indicators.vwap", return_value=pd.Series([])):
            self._run_upgrade(mgr, "test-001", provider)  # must not raise

        self.assertIsNone(st.or_high)

    def test_snapshot_only_symbols_unchanged(self) -> None:
        """The watchlist block must not touch snapshot_only seeds (they have their own path)."""
        mgr = _make_manager()
        sess = _make_session()
        st = _make_watchlist_state("SNAP")
        st.seed_source = "snapshot_only"  # different seed
        sess.symbols["SNAP"] = st
        mgr._sessions["test-001"] = sess

        provider = MagicMock()
        provider.get_bars_range.return_value = _fake_bars(10)
        mgr._provider = provider

        # Patch _build_seed_from_provider to prevent the snapshot upgrade path from running
        with patch.object(mgr, "_build_seed_from_provider", side_effect=Exception("no build")):
            with patch("scanner.indicators.vwap", return_value=pd.Series([150.0] * 10)):
                # Will raise in snapshot path for snapshot_only but watchlist block should not touch it
                try:
                    self._run_upgrade(mgr, "test-001", provider)
                except Exception:
                    pass

        # or_high should still be None — watchlist block doesn't touch snapshot_only
        self.assertIsNone(st.or_high)


if __name__ == "__main__":
    unittest.main()
