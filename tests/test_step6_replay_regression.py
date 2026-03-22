from __future__ import annotations

import unittest
from unittest.mock import patch

from monitor.live_monitor import LiveMonitorManager
from scanner import orb
from tests.replay_loader import (
    build_provider_from_fixture,
    fixture_symbols,
    frozen_datetime_from_fixture,
    load_saved_snapshot,
    monitor_now_from_fixture,
)
from tests.test_characterization_suite import _monitor_snapshot, _scan_cfg, _scan_snapshot


class Step6ReplayRegressionTests(unittest.TestCase):
    def _run_scan(self, fixture_name: str):
        provider, fixture = build_provider_from_fixture(fixture_name)
        with patch.object(orb, "datetime", frozen_datetime_from_fixture(fixture)):
            result = orb.scan_symbols(
                fixture_symbols(fixture),
                _scan_cfg(),
                limit=10,
                provider=provider,
                use_ml=False,
                use_sentiment=False,
                use_catalyst=False,
                long_only=True,
                monitor_first_mode=True,
                min_combined_enabled=False,
                min_grade_enabled=False,
                min_vwap_enabled=False,
                no_chop_enabled=False,
                orb_retest_min_ml_score=0.0,
            )
        return provider, fixture, result

    def _run_monitor_snapshot(self, fixture_name: str) -> dict:
        provider, fixture, result = self._run_scan(fixture_name)
        with patch.object(LiveMonitorManager, "_ensure_worker", lambda self: None):
            manager = LiveMonitorManager()
            with patch("monitor.live_monitor._now_ts", lambda: monitor_now_from_fixture(fixture)):
                session = manager.start_from_scan_candidates(
                    job_id=f"replay-{fixture_name}",
                    candidates=(result.get("seed_candidates") or result["candidates"]),
                    top_n=5,
                    provider=provider,
                    stream_cache=None,
                    long_only=True,
                )
                status = manager.status(session.monitor_id, provider=provider, stream_cache=None, refresh=True)
                replay = manager.replay(monitor_id=session.monitor_id)
        return _monitor_snapshot(status, replay)

    def test_nominal_replay_scan_matches_saved_snapshot(self) -> None:
        _, _, result = self._run_scan("orb_nominal_replay.json")
        self.assertEqual(
            _scan_snapshot(result),
            load_saved_snapshot("step6_nominal_scan_snapshot.json"),
        )

    def test_nominal_replay_monitor_matches_saved_snapshot(self) -> None:
        self.assertEqual(
            self._run_monitor_snapshot("orb_nominal_replay.json"),
            load_saved_snapshot("step6_nominal_monitor_snapshot.json"),
        )

    def test_preopen_replay_resolves_prior_session_date(self) -> None:
        provider, fixture = build_provider_from_fixture("orb_preopen_replay.json")
        with patch.object(orb, "datetime", frozen_datetime_from_fixture(fixture)):
            resolved = orb.resolve_session_date(provider, probe_symbol="AAPL")
        self.assertEqual(
            {"resolved_session_date": resolved.isoformat()},
            load_saved_snapshot("step6_preopen_session_snapshot.json"),
        )

    def test_dead_tape_monitor_matches_saved_snapshot(self) -> None:
        self.assertEqual(
            self._run_monitor_snapshot("orb_dead_tape_replay.json"),
            load_saved_snapshot("step6_dead_tape_monitor_snapshot.json"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
