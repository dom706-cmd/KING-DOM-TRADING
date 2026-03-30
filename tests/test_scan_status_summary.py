from __future__ import annotations

import unittest

import app as kingdom_app


class ScanStatusSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = kingdom_app.app.test_client()
        self._jobs_snapshot = dict(kingdom_app._JOBS)
        self._apply_snapshot = kingdom_app._apply_live_state_to_candidates
        kingdom_app._apply_live_state_to_candidates = lambda rows: list(rows or [])

    def tearDown(self) -> None:
        kingdom_app._apply_live_state_to_candidates = self._apply_snapshot
        kingdom_app._JOBS.clear()
        kingdom_app._JOBS.update(self._jobs_snapshot)

    def _seed_done_job(self) -> str:
        jid = "scan-status-fallback"
        kingdom_app._JOBS[jid] = {
            "status": "done",
            "started_at": 0.0,
            "updated_at": 0.0,
            "progress": {"scanned": 400, "chunks_done": 4, "chunks_total": 4, "offset": 0, "end_offset": 400},
            "result": {
                "provider": "alpaca",
                "strategy": "orb",
                "count": 0,
                "candidates_total": 0,
                "tradable_now_total": 0,
                "seed_candidates_total": 2,
                "rejected_total": 104,
                "candidates": [],
                "seed_candidates": [
                    {"symbol": "PLTR", "combined_score": 0.82, "monitor_seed_reasons": ["orb_retest_too_extended"]},
                    {"symbol": "NVDA", "combined_score": 0.91, "monitor_seed_reasons": ["orb_retest_not_near_entry"]},
                ],
                "reject_counts": {"filtered_rvol": 40, "orb_retest_too_extended": 17},
                "shortlisted": 104,
                "scanned": 400,
                "chunks": 4,
                "end_offset": 400,
                "universe_size": 400,
                "thresholds_used": {"limit": 25},
                "zero_result_diagnostics": {},
                "primary_candidates": [],
                "primary_mode": None,
                "primary_message": None,
            },
            "partial_result": {},
            "error": None,
            "params": {},
            "thresholds_used": {"limit": 25},
            "provider": "alpaca",
        }
        return jid

    def test_scan_status_rebuilds_primary_and_fallback_rows(self) -> None:
        jid = self._seed_done_job()
        response = self.client.get(f"/api/scan_status?job_id={jid}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        result = payload["result"]
        self.assertEqual(result["primary_mode"], "monitor_watch")
        self.assertIn("Showing the best monitor/watch seeds", str(result["primary_message"]))
        self.assertEqual([row["symbol"] for row in result["primary_candidates"]], ["NVDA", "PLTR"])
        self.assertEqual([row["symbol"] for row in result["fallback_candidates"]], ["NVDA", "PLTR"])
        self.assertEqual(result["trade_ready_candidate_count"], 0)
        self.assertEqual(result["fallback_candidate_count"], 2)
        self.assertTrue(result["zero_result_diagnostics"]["monitor_fallback_active"])
        self.assertEqual(result["zero_result_diagnostics"]["primary_symbols"], ["NVDA", "PLTR"])

    def test_debug_last_scan_rebuilds_primary_and_fallback_rows(self) -> None:
        jid = self._seed_done_job()
        response = self.client.get(f"/api/debug_last_scan?job_id={jid}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["primary_mode"], "monitor_watch")
        self.assertEqual([row["symbol"] for row in payload["primary_candidates"]], ["NVDA", "PLTR"])
        self.assertEqual([row["symbol"] for row in payload["fallback_candidates"]], ["NVDA", "PLTR"])
        self.assertEqual(payload["trade_ready_candidate_count"], 0)
        self.assertEqual(payload["fallback_candidate_count"], 2)
        self.assertTrue(payload["zero_result_diagnostics"]["monitor_fallback_active"])

    def test_scan_status_uses_partial_result_when_job_errors(self) -> None:
        jid = "scan-status-error-fallback"
        kingdom_app._JOBS[jid] = {
            "status": "error",
            "started_at": 0.0,
            "updated_at": 0.0,
            "progress": {"scanned": 200, "chunks_done": 2, "chunks_total": 4, "offset": 0, "end_offset": 400},
            "result": None,
            "partial_result": {
                "provider": "alpaca",
                "strategy": "orb",
                "candidates_total": 0,
                "tradable_now_total": 0,
                "seed_candidates_total": 1,
                "rejected_total": 17,
                "seed_candidates": [{"symbol": "AMD", "combined_score": 0.77}],
                "thresholds_used": {"limit": 25},
            },
            "error": "live_provider_unavailable",
            "params": {},
            "thresholds_used": {"limit": 25},
            "provider": "alpaca",
        }
        response = self.client.get(f"/api/scan_status?job_id={jid}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "live_provider_unavailable")
        result = payload["result"]
        self.assertEqual(result["primary_mode"], "monitor_watch")
        self.assertEqual([row["symbol"] for row in result["primary_candidates"]], ["AMD"])
        self.assertEqual([row["symbol"] for row in result["fallback_candidates"]], ["AMD"])
        self.assertEqual(result["fallback_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
