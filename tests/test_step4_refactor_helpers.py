from __future__ import annotations

import unittest
from types import SimpleNamespace

from monitor import live_monitor
from scanner import orb


class Step4RefactorHelperTests(unittest.TestCase):
    def test_orb_enrichment_ranker_prefers_ml_then_liquidity(self) -> None:
        a = orb.Candidate(symbol='A', data_date='2026-03-10', last_price=10.0, pct_change=None, rvol=2.0, today_dollar_vol=1_000_000.0, avg20_dollar_vol=2_000_000.0, or_high=10.5, or_low=9.8, or_range_pct=1.2, above_vwap=True, vwap_last=10.1, vwap_delta_pct=0.5, trend_state='up', trend_slope_pct=0.2, best_side='long', entry=10.2, stop=9.9, target_2r=10.8, target_3r=11.1, risk_per_share=0.3, shares=100, notional=1020.0, long_entry=10.2, long_stop=9.9, long_2r=10.8, long_3r=11.1, long_risk_per_share=0.3, long_shares=100, long_notional=1020.0, short_entry=None, short_stop=None, short_2r=None, short_3r=None, short_risk_per_share=None, short_shares=None, short_notional=None)
        b = orb.Candidate(symbol='B', data_date='2026-03-10', last_price=11.0, pct_change=None, rvol=5.0, today_dollar_vol=5_000_000.0, avg20_dollar_vol=3_000_000.0, or_high=11.4, or_low=10.7, or_range_pct=1.5, above_vwap=True, vwap_last=10.9, vwap_delta_pct=0.8, trend_state='up', trend_slope_pct=0.3, best_side='long', entry=11.1, stop=10.8, target_2r=11.7, target_3r=12.0, risk_per_share=0.3, shares=100, notional=1110.0, long_entry=11.1, long_stop=10.8, long_2r=11.7, long_3r=12.0, long_risk_per_share=0.3, long_shares=100, long_notional=1110.0, short_entry=None, short_stop=None, short_2r=None, short_3r=None, short_risk_per_share=None, short_shares=None, short_notional=None)
        a.ml_score = 0.90
        b.ml_score = 0.40
        ranked = orb._rank_candidates_for_optional_enrichment([b, a], use_ml=True)
        self.assertEqual([c.symbol for c in ranked], ['A', 'B'])

    def test_orb_base_score_prefers_better_activity(self) -> None:
        slow = orb.Candidate(symbol='S', data_date='2026-03-10', last_price=10.0, pct_change=None, rvol=0.5, today_dollar_vol=100_000.0, avg20_dollar_vol=1_000_000.0, or_high=10.2, or_low=9.9, or_range_pct=0.8, above_vwap=False, vwap_last=10.0, vwap_delta_pct=0.0, trend_state='chop', trend_slope_pct=0.0, best_side='long', entry=10.1, stop=9.9, target_2r=10.5, target_3r=10.7, risk_per_share=0.2, shares=100, notional=1010.0, long_entry=10.1, long_stop=9.9, long_2r=10.5, long_3r=10.7, long_risk_per_share=0.2, long_shares=100, long_notional=1010.0, short_entry=None, short_stop=None, short_2r=None, short_3r=None, short_risk_per_share=None, short_shares=None, short_notional=None)
        fast = orb.Candidate(symbol='F', data_date='2026-03-10', last_price=10.0, pct_change=None, rvol=4.0, today_dollar_vol=4_000_000.0, avg20_dollar_vol=1_000_000.0, or_high=10.3, or_low=9.8, or_range_pct=2.4, above_vwap=True, vwap_last=9.95, vwap_delta_pct=0.5, trend_state='up', trend_slope_pct=0.2, best_side='long', entry=10.05, stop=9.85, target_2r=10.45, target_3r=10.65, risk_per_share=0.2, shares=100, notional=1005.0, long_entry=10.05, long_stop=9.85, long_2r=10.45, long_3r=10.65, long_risk_per_share=0.2, long_shares=100, long_notional=1005.0, short_entry=None, short_stop=None, short_2r=None, short_3r=None, short_risk_per_share=None, short_shares=None, short_notional=None)
        self.assertGreater(orb._orb_base_score(fast), orb._orb_base_score(slow))

    def test_monitor_live_score_helper_matches_expected_formula(self) -> None:
        st = SimpleNamespace(tape_live=True, catalyst_score=0.5, catalyst_freshness_hours=2.0)

        class PB:
            def score(self, st, live_state, context):
                return 7.25

        score = live_monitor._compute_monitor_live_score(
            st,
            playbook=PB(),
            live_state={},
            context={},
            context_score=0.0,
            flags=[],
        )
        self.assertAlmostEqual(score, 26.25, places=6)

    def test_monitor_confidence_reasons_add_live_and_vwap_alignment(self) -> None:
        st = SimpleNamespace(tape_live=True, above_vwap_live=True, best_side='long', catalyst_freshness_hours=1.0)
        reasons = live_monitor._monitor_live_confidence_reasons(st, ['arming'])
        self.assertEqual(reasons, ['arming', 'tape_live', 'vwap_aligned_long', 'news_fresh'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
