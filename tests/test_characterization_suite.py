from __future__ import annotations

import json
import sys
import types
import unittest
from datetime import date, datetime as real_datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_optional_dependency_stubs() -> None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401
    except Exception:
        pkg = types.ModuleType("vaderSentiment")
        sub = types.ModuleType("vaderSentiment.vaderSentiment")

        class SentimentIntensityAnalyzer:  # type: ignore[override]
            def polarity_scores(self, text: str) -> dict[str, float]:
                return {"compound": 0.0}

        sub.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer
        sys.modules.setdefault("vaderSentiment", pkg)
        sys.modules["vaderSentiment.vaderSentiment"] = sub


_install_optional_dependency_stubs()

from monitor.live_monitor import LiveMonitorManager
from providers.base import BarsRequest
from scanner import orb

ET = ZoneInfo("America/New_York")
SNAPSHOT_DIR = Path(__file__).resolve().parent / "fixtures" / "snapshots"
FIXTURE_SESSION_DATE = date(2026, 3, 10)


class ReplayProvider:
    name = "replay"

    def __init__(self, *, daily: dict[str, pd.DataFrame], intraday: dict[str, pd.DataFrame], latest_trade: dict[str, dict], latest_quote: dict[str, dict]) -> None:
        self._daily = {str(k).upper(): v.copy() for k, v in daily.items()}
        self._intraday = {str(k).upper(): v.copy() for k, v in intraday.items()}
        self._latest_trade = {str(k).upper(): dict(v) for k, v in latest_trade.items()}
        self._latest_quote = {str(k).upper(): dict(v) for k, v in latest_quote.items()}

    def download_daily_batch(self, symbols: list[str], period: str = "3mo") -> dict[str, pd.DataFrame]:
        return {str(symbol).upper(): self._daily.get(str(symbol).upper(), pd.DataFrame()).copy() for symbol in symbols}

    def get_daily_history(self, symbol: str, period: str = "6mo", timeout_s: int | None = None) -> pd.DataFrame:
        return self._daily.get(str(symbol).upper(), pd.DataFrame()).copy()

    def get_bars_range(
        self,
        *,
        symbol: str,
        interval: str,
        from_d: date,
        to_d: date,
        include_prepost: bool = False,
        timeout_s: int | None = None,
    ) -> pd.DataFrame:
        return self._intraday.get(str(symbol).upper(), pd.DataFrame()).copy()

    def get_bars(self, req: BarsRequest, timeout_s: int | None = None) -> pd.DataFrame:
        return self._intraday.get(str(req.symbol).upper(), pd.DataFrame()).copy()

    def get_latest_trade(self, symbol: str) -> dict:
        return dict(self._latest_trade[str(symbol).upper()])

    def get_latest_trade_price(self, symbol: str) -> float:
        return float(self._latest_trade[str(symbol).upper()]["price"])

    def get_latest_quote(self, symbol: str) -> dict:
        return dict(self._latest_quote[str(symbol).upper()])

    def get_news(self, *args, **kwargs) -> list[dict]:
        return []


def _build_replay_provider() -> ReplayProvider:
    dates = pd.bdate_range("2026-02-03", "2026-03-09", tz=ET)

    daily_abcd = pd.DataFrame(
        {
            "Open": [11.0 + 0.05 * i for i in range(len(dates))],
            "High": [11.2 + 0.05 * i for i in range(len(dates))],
            "Low": [10.9 + 0.05 * i for i in range(len(dates))],
            "Close": [11.1 + 0.05 * i for i in range(len(dates))],
            "Volume": [320_000 + (i % 5) * 15_000 for i in range(len(dates))],
        },
        index=dates,
    )

    daily_thin = pd.DataFrame(
        {
            "Open": [8.0 + 0.01 * i for i in range(len(dates))],
            "High": [8.1 + 0.01 * i for i in range(len(dates))],
            "Low": [7.9 + 0.01 * i for i in range(len(dates))],
            "Close": [8.0 + 0.01 * i for i in range(len(dates))],
            "Volume": [10_000 + (i % 3) * 1_000 for i in range(len(dates))],
        },
        index=dates,
    )

    intraday_index = pd.date_range("2026-03-10 09:30", periods=31, freq="1min", tz=ET)
    intraday_abcd = pd.DataFrame(
        [
            (12.00, 12.05, 11.95, 12.02, 80_000),
            (12.02, 12.08, 11.98, 12.06, 85_000),
            (12.06, 12.10, 12.01, 12.03, 90_000),
            (12.03, 12.07, 11.97, 11.99, 82_000),
            (11.99, 12.04, 11.90, 11.96, 95_000),
            (11.96, 12.09, 11.95, 12.08, 100_000),
            (12.08, 12.16, 12.05, 12.14, 110_000),
            (12.14, 12.18, 12.09, 12.11, 105_000),
            (12.11, 12.13, 12.06, 12.07, 98_000),
            (12.07, 12.12, 12.04, 12.10, 97_000),
            (12.10, 12.15, 12.09, 12.14, 93_000),
            (12.14, 12.19, 12.12, 12.18, 92_000),
            (12.18, 12.24, 12.15, 12.22, 91_000),
            (12.22, 12.27, 12.18, 12.20, 90_000),
            (12.20, 12.23, 12.16, 12.17, 89_000),
            (12.17, 12.21, 12.14, 12.19, 88_000),
            (12.19, 12.26, 12.18, 12.25, 87_000),
            (12.25, 12.30, 12.21, 12.28, 86_000),
            (12.28, 12.34, 12.26, 12.33, 85_000),
            (12.33, 12.36, 12.29, 12.31, 84_000),
            (12.31, 12.35, 12.28, 12.34, 83_000),
            (12.34, 12.38, 12.32, 12.37, 82_000),
            (12.37, 12.39, 12.33, 12.35, 81_000),
            (12.35, 12.40, 12.34, 12.39, 80_000),
            (12.39, 12.44, 12.38, 12.42, 79_000),
            (12.42, 12.46, 12.39, 12.41, 78_000),
            (12.41, 12.47, 12.40, 12.45, 77_000),
            (12.45, 12.49, 12.43, 12.48, 76_000),
            (12.48, 12.50, 12.44, 12.46, 75_000),
            (12.46, 12.53, 12.45, 12.51, 74_000),
            (12.51, 12.56, 12.49, 12.54, 73_000),
        ],
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=intraday_index,
    )

    latest_trade = {
        "ABCD": {"price": 12.54, "timestamp": "2026-03-10T14:00:00+00:00"},
        "AAPL": {"price": 12.54, "timestamp": "2026-03-10T14:00:00+00:00"},
    }
    latest_quote = {
        "ABCD": {
            "bid_price": 12.53,
            "ask_price": 12.55,
            "bid_size": 10,
            "ask_size": 12,
            "timestamp": "2026-03-10T14:00:00+00:00",
        },
        "AAPL": {
            "bid_price": 12.53,
            "ask_price": 12.55,
            "bid_size": 10,
            "ask_size": 12,
            "timestamp": "2026-03-10T14:00:00+00:00",
        },
    }

    return ReplayProvider(
        daily={"ABCD": daily_abcd, "THIN": daily_thin, "AAPL": daily_abcd},
        intraday={"ABCD": intraday_abcd, "AAPL": intraday_abcd},
        latest_trade=latest_trade,
        latest_quote=latest_quote,
    )


def _scan_cfg() -> orb.ORBConfig:
    return orb.ORBConfig(
        min_price=1.0,
        max_price=30.0,
        min_today_dollar_vol=500_000.0,
        min_avg20_dollar_vol=1_000_000.0,
        min_rvol=0.5,
        min_or_range_pct=0.0,
        max_or_range_pct=10.0,
        max_risk_per_share=5.0,
    )


def _frozen_orb_datetime(ts: str):
    frozen = pd.Timestamp(ts, tz=ET).to_pydatetime()

    class FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return frozen

        @classmethod
        def combine(cls, *args, **kwargs):
            return real_datetime.combine(*args, **kwargs)

    return FrozenDateTime


def _monitor_now_ts() -> float:
    return pd.Timestamp("2026-03-10 10:00:00", tz=ET).timestamp()


def _round_float(value: float | None, digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits)


def _load_snapshot(name: str) -> dict:
    return json.loads((SNAPSHOT_DIR / name).read_text())


def _plan_snapshot(candidate) -> dict:
    return {
        "symbol": candidate.symbol,
        "best_side": candidate.best_side,
        "entry": _round_float(candidate.entry, 6),
        "stop": _round_float(candidate.stop, 6),
        "target_2r": _round_float(candidate.target_2r, 6),
        "target_3r": _round_float(candidate.target_3r, 6),
        "last_price": _round_float(candidate.last_price, 6),
        "rvol": _round_float(candidate.rvol, 6),
        "today_dollar_vol": _round_float(candidate.today_dollar_vol, 6),
        "avg20_dollar_vol": _round_float(candidate.avg20_dollar_vol, 6),
        "orb_retest_ready": bool(candidate.orb_retest_ready),
        "orb_retest_distance_r": _round_float(candidate.orb_retest_distance_r, 6),
        "tradable_now": bool(candidate.tradable_now),
    }


def _scan_snapshot(result: dict) -> dict:
    effective = (result.get("candidates") or result.get("seed_candidates") or [])
    top = effective[0]
    return {
        "count": int(result.get("count") if result.get("candidates") else len(effective)),
        "candidates_total": int(result.get("candidates_total") if result.get("candidates") else len(effective)),
        "seed_candidates_total": int(result["seed_candidates_total"]),
        "tradable_now_total": int(result["tradable_now_total"]),
        "rejected_total": int(result["rejected_total"]),
        "scan_date": result["scan_date"],
        "prefilter_counts": dict(result["prefilter_counts"]),
        "reject_counts_nonzero": {k: v for k, v in dict(result["reject_counts"]).items() if v},
        "top_candidate": {
            "symbol": top["symbol"],
            "best_side": top["best_side"],
            "entry": _round_float(top.get("entry"), 6),
            "stop": _round_float(top.get("stop"), 6),
            "target_2r": _round_float(top.get("target_2r"), 6),
            "target_3r": _round_float(top.get("target_3r"), 6),
            "last_price": _round_float(top.get("last_price"), 6),
            "rvol": _round_float(top.get("rvol"), 6),
            "today_dollar_vol": _round_float(top.get("today_dollar_vol"), 6),
            "avg20_dollar_vol": _round_float(top.get("avg20_dollar_vol"), 6),
            "combined_score": _round_float(top.get("combined_score"), 6),
            "confidence_score": _round_float(top.get("confidence_score"), 6),
            "confidence_grade": top.get("confidence_grade"),
            "gate_passes": bool(top.get("gate_passes")),
            "gate_fail_reasons": list(top.get("gate_fail_reasons") or []),
            "orb_retest_ready": bool(top.get("orb_retest_ready")),
            "orb_retest_distance_r": _round_float(top.get("orb_retest_distance_r"), 6),
            "monitor_seed": bool(top.get("monitor_seed")),
            "tradable_now": bool(top.get("tradable_now")),
        },
    }


def _monitor_snapshot(status: dict, replay: dict) -> dict:
    top_symbol = status["symbols"][0]
    top_live = status["summary"]["top_live_symbol"] or {}
    return {
        "ready_count": int(status["summary"]["ready_count"]),
        "live_triggered_count": int(status["summary"]["live_triggered_count"]),
        "top_live_symbol": {
            "symbol": top_live.get("symbol"),
            "live_score": _round_float(top_live.get("live_score"), 6),
            "state": top_live.get("state"),
        },
        "symbol": top_symbol["symbol"],
        "monitor_state": top_symbol["monitor_state"],
        "tape_live": bool(top_symbol["tape_live"]),
        "tape_live_reason": top_symbol["tape_live_reason"],
        "spread_pct": _round_float(top_symbol["spread_pct"], 6),
        "live_score": _round_float(top_symbol["live_score"], 6),
        "live_confidence_grade": top_symbol["live_confidence_grade"],
        "live_confidence_score": _round_float(top_symbol["live_confidence_score"], 6),
        "live_confidence_reasons": list(top_symbol["live_confidence_reasons"]),
        "replay_ok": bool(replay["ok"]),
        "replay_symbol_count": len(replay["replay"]["session"]["symbols"]),
    }


def _run_scan(provider: ReplayProvider) -> dict:
    with patch.object(orb, "datetime", _frozen_orb_datetime("2026-03-10 10:05:00")):
        return orb.scan_symbols(
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


class CharacterizationSuite(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.provider = _build_replay_provider()

    def test_resolve_session_date_prefers_today_after_open_when_probe_has_today_bars(self) -> None:
        with patch.object(orb, "datetime", _frozen_orb_datetime("2026-03-10 10:05:00")):
            resolved = orb.resolve_session_date(self.provider, probe_symbol="AAPL")
        self.assertEqual(resolved.isoformat(), "2026-03-10")

    def test_resolve_session_date_uses_prior_weekday_before_open(self) -> None:
        with patch.object(orb, "datetime", _frozen_orb_datetime("2026-03-10 09:15:00")):
            resolved = orb.resolve_session_date(self.provider, probe_symbol="AAPL")
        self.assertEqual(resolved.isoformat(), "2026-03-09")

    def test_build_orb_plan_matches_characterization_snapshot(self) -> None:
        with patch.object(orb, "datetime", _frozen_orb_datetime("2026-03-10 10:05:00")):
            candidate = orb.build_orb_plan(self.provider, "ABCD", _scan_cfg(), session_date=FIXTURE_SESSION_DATE)
        self.assertEqual(_plan_snapshot(candidate), _load_snapshot("orb_build_plan_snapshot.json"))

    def test_scan_symbols_matches_characterization_snapshot(self) -> None:
        result = _run_scan(self.provider)
        self.assertEqual(_scan_snapshot(result), _load_snapshot("orb_scan_snapshot.json"))

    def test_live_monitor_status_matches_characterization_snapshot(self) -> None:
        result = _run_scan(self.provider)
        with patch.object(LiveMonitorManager, "_ensure_worker", lambda self: None):
            manager = LiveMonitorManager()
            with patch("monitor.live_monitor._now_ts", _monitor_now_ts):
                session = manager.start_from_scan_candidates(
                    job_id="job-1",
                    candidates=(result.get("seed_candidates") or result["candidates"]),
                    top_n=5,
                    provider=self.provider,
                    stream_cache=None,
                    long_only=True,
                )
                status = manager.status(session.monitor_id, provider=self.provider, stream_cache=None, refresh=True)
                replay = manager.replay(monitor_id=session.monitor_id)
        self.assertEqual(_monitor_snapshot(status, replay), _load_snapshot("live_monitor_snapshot.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
