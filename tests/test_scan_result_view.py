from __future__ import annotations

import unittest

from scanner.result_view import build_primary_fallback_view, build_zero_result_diagnostics, select_primary_candidates


class ScanResultViewTests(unittest.TestCase):
    def test_primary_candidates_prefer_trade_ready_rows(self) -> None:
        rows, mode, message = select_primary_candidates(
            {
                "candidates": [{"symbol": "AAPL", "combined_score": 0.9}],
                "seed_candidates": [{"symbol": "MSFT", "combined_score": 0.95}],
            }
        )
        self.assertEqual(mode, "trade_ready")
        self.assertIsNone(message)
        self.assertEqual([row["symbol"] for row in rows], ["AAPL"])

    def test_primary_candidates_fall_back_to_seed_rows_when_trade_ready_empty(self) -> None:
        rows, mode, message = select_primary_candidates(
            {
                "candidates": [],
                "seed_candidates": [
                    {"symbol": "MSFT", "combined_score": 0.75},
                    {"symbol": "AAPL", "combined_score": 0.95},
                ],
            }
        )
        self.assertEqual(mode, "monitor_watch")
        self.assertIn("Showing the best monitor/watch seeds", str(message))
        self.assertEqual([row["symbol"] for row in rows], ["AAPL", "MSFT"])

    def test_zero_result_diagnostics_reports_monitor_fallback(self) -> None:
        diagnostics = build_zero_result_diagnostics(
            {
                "candidates_total": 0,
                "seed_candidates_total": 2,
                "rejected_total": 7,
                "shortlisted": 9,
                "seed_candidates": [{"symbol": "PLTR"}, {"symbol": "NVDA"}],
                "reject_counts": {"filtered_avg20_dollar_vol": 11, "orb_retest_too_extended": 4},
                "prefilter_counts": {"daily_ok": 12},
            }
        )
        self.assertTrue(diagnostics["trade_ready_empty"])
        self.assertTrue(diagnostics["monitor_fallback_active"])
        self.assertEqual(diagnostics["primary_mode"], "monitor_watch")
        self.assertEqual(diagnostics["primary_symbols"], ["PLTR", "NVDA"])
        self.assertEqual(diagnostics["top_rejection_reasons"][0], ("filtered_avg20_dollar_vol", 11))

    def test_primary_fallback_view_surfaces_seed_rows_when_trade_ready_is_empty(self) -> None:
        view = build_primary_fallback_view(
            {
                "candidates_total": 0,
                "tradable_now_total": 0,
                "seed_candidates_total": 2,
                "seed_candidates": [
                    {"symbol": "PLTR", "combined_score": 0.84},
                    {"symbol": "NVDA", "combined_score": 0.91},
                ],
            }
        )
        self.assertEqual(view["primary_mode"], "monitor_watch")
        self.assertEqual([row["symbol"] for row in view["primary_candidates"]], ["NVDA", "PLTR"])
        self.assertEqual([row["symbol"] for row in view["fallback_candidates"]], ["NVDA", "PLTR"])
        self.assertEqual(view["trade_ready_candidate_count"], 0)
        self.assertEqual(view["fallback_candidate_count"], 2)
        self.assertTrue(view["zero_result_diagnostics"]["monitor_fallback_active"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
