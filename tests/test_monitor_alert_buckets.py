from __future__ import annotations

import unittest
from types import SimpleNamespace

from monitor import live_monitor


class MonitorAlertBucketTests(unittest.TestCase):
    def test_alert_bucket_distinguishes_near_ready_and_triggered(self) -> None:
        near = SimpleNamespace(
            tape_live=True,
            entry=10.0,
            risk_per_share=0.5,
            price=9.95,
            retest_distance_r=0.10,
            long_triggered_live=False,
            short_triggered_live=False,
        )
        self.assertEqual(live_monitor._alert_bucket_for_state(near, 'watch'), 'near_trigger')
        self.assertEqual(live_monitor._alert_bucket_for_state(near, 'arming'), 'ready')
        self.assertEqual(live_monitor._alert_bucket_for_state(near, 'triggered'), 'triggered')


if __name__ == '__main__':
    unittest.main(verbosity=2)
