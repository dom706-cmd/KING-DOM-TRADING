from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import app as app_module
from core.errors import IntradayDataFailure, TrendContextFailure, MonitorTradeRefreshFailure
from monitor import live_monitor
from scanner import orb


class Step5TypedFailuresTests(unittest.TestCase):
    def test_orb_build_failure_payload_is_typed_and_keeps_context(self) -> None:
        payload = orb._orb_build_failure_payload(
            sym="ABCD",
            err=ValueError("intraday_empty"),
            session_date=date(2026, 3, 10),
            strategy="orb",
            exec_style="retest",
        )
        self.assertEqual(payload["symbol"], "ABCD")
        self.assertEqual(payload["stage"], "intraday")
        self.assertIn("code", payload)
        self.assertIn("message", payload)
        self.assertEqual(payload["context"]["session_date"], "2026-03-10")
        self.assertEqual(payload["context"]["strategy"], "orb")

    def test_entry_now_intraday_context_raises_typed_failure_for_empty_intraday(self) -> None:
        class DummyProvider:
            def get_bars(self, req):
                return pd.DataFrame()

        with self.assertRaises(IntradayDataFailure) as cm:
            app_module._entry_now_intraday_context(DummyProvider(), "ABCD", include_prepost=False)

        self.assertEqual(cm.exception.code, "intraday_empty")
        self.assertEqual(cm.exception.symbol, "ABCD")

    def test_entry_now_trend_context_raises_typed_failure(self) -> None:
        intraday = pd.DataFrame(
            {"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.0], "Volume": [1000]},
            index=pd.date_range("2026-03-10 09:30", periods=1, freq="1min"),
        )

        with patch("scanner.indicators.vwap", side_effect=RuntimeError("boom")):
            with self.assertRaises(TrendContextFailure) as cm:
                app_module._entry_now_trend_context(intraday)

        self.assertEqual(cm.exception.code, "trend_context_failed")

    def test_monitor_trade_refresh_update_raises_typed_failure(self) -> None:
        class DummyProvider:
            def get_latest_trade(self, sym):
                raise RuntimeError("provider_down")

        class DummyStream:
            def latest_trade(self, sym):
                raise ValueError("stream_bad")

        with self.assertRaises(MonitorTradeRefreshFailure) as cm:
            live_monitor._trade_refresh_update("ABCD", provider=DummyProvider(), stream_cache=DummyStream())

        self.assertEqual(cm.exception.code, "trade_refresh_failed")
        self.assertIn("stream=", cm.exception.message)
        self.assertIn("provider=", cm.exception.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
