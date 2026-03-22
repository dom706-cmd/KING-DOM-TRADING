from __future__ import annotations

import unittest
from unittest.mock import patch

from scanner import orb
from tests.test_characterization_suite import _build_replay_provider, _frozen_orb_datetime, _scan_cfg


class OrbDiscoveryTradeReadySplitTests(unittest.TestCase):
    def test_monitor_seed_can_pass_discovery_without_being_trade_ready(self) -> None:
        provider = _build_replay_provider()
        with patch.object(orb, "datetime", _frozen_orb_datetime("2026-03-10 10:05:00")):
            result = orb.scan_symbols(
                ["ABCD", "THIN"],
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

        self.assertEqual(int(result["discovery_total"]), 1)
        self.assertEqual(int(result["seed_candidates_total"]), 1)
        self.assertEqual(int(result["tradable_now_total"]), 0)
        self.assertEqual(int(result["trade_ready_total"]), 0)

        top = result["seed_candidates"][0]
        self.assertTrue(bool(top["discovery_passes"]))
        self.assertFalse(bool(top["trade_ready_passes"]))
        self.assertTrue(bool(top["monitor_seed"]))
        self.assertFalse(bool(top["tradable_now"]))
        self.assertFalse(bool(top["gate_passes"]))

        self.assertIn("chase_too_high", list(top["gate_fail_reasons"] or []))
        self.assertIn("orb_retest_too_extended", list(top["gate_fail_reasons"] or []))
        self.assertIn("chase_too_high", list(top["trade_ready_fail_reasons"] or []))
        self.assertIn("orb_retest_too_extended", list(top["trade_ready_fail_reasons"] or []))

        discovery_reasons = list(top["discovery_fail_reasons"] or [])
        self.assertEqual(discovery_reasons, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
