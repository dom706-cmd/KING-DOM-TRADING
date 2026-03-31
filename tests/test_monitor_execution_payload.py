from __future__ import annotations

import unittest
from unittest.mock import patch

from monitor.live_monitor import LiveMonitorManager
from tests.replay_loader import build_provider_from_fixture, monitor_now_from_fixture
from tests.test_characterization_suite import _run_scan


class MonitorExecutionPayloadTests(unittest.TestCase):
    def test_monitor_status_exposes_execution_plan_and_rejections(self) -> None:
        provider, fixture = build_provider_from_fixture("orb_nominal_replay.json")
        result = _run_scan(provider)
        candidates = list(result.get("seed_candidates") or result["candidates"])

        with patch.object(LiveMonitorManager, "_ensure_worker", lambda self: None):
            manager = LiveMonitorManager()
            with patch("monitor.live_monitor._now_ts", lambda: monitor_now_from_fixture(fixture)):
                session = manager.start_from_scan_candidates(
                    job_id="job-execution-payload",
                    candidates=candidates,
                    top_n=5,
                    provider=provider,
                    stream_cache=None,
                    long_only=True,
                )
                status = manager.status(session.monitor_id, provider=provider, stream_cache=None, refresh=True)

        symbol = status["symbols"][0]
        self.assertEqual(symbol["symbol"], "ABCD")
        self.assertEqual(symbol["plan_state"], "WAIT")
        self.assertIn("execution_plan", symbol)
        self.assertIn("tradeability", symbol)
        self.assertEqual(symbol["execution_plan"]["stop_loss"], symbol["stop_loss"])
        self.assertEqual(symbol["execution_plan"]["target_2r"], symbol["target_2r"])
        self.assertEqual(symbol["execution_plan"]["target_3r"], symbol["target_3r"])
        self.assertFalse(bool(symbol["execution_plan"]["tradable_now"]))
        self.assertFalse(bool(symbol["execution_plan"]["orb_retest_ready"]))
        self.assertIn("chase_too_high", list(symbol["rejection_reasons"]))
        self.assertIn("orb_retest_too_extended", list(symbol["rejection_reasons"]))
        self.assertEqual(symbol["tradeability"]["tape_live_reason"], symbol["tape_live_reason"])
        self.assertEqual(symbol["tradeability"]["spread_pct"], symbol["spread_pct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
