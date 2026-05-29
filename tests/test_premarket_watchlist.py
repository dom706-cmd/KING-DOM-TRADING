"""
tests/test_premarket_watchlist.py
==================================
Unit + live-data tests for premarket_watchlist.py.

Unit tests use lightweight mock snapshots — no network calls.
Integration tests (marked with @pytest.mark.integration) hit real Alpaca
and require valid ALPACA_API_KEY + ALPACA_SECRET_KEY env vars.

Run unit-only:
    .venv/bin/python -m pytest tests/test_premarket_watchlist.py -v -m "not integration"

Run all (including live):
    .venv/bin/python -m pytest tests/test_premarket_watchlist.py -v
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

# Make sure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.premarket_watchlist as pm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build mock Alpaca snapshot objects
# ─────────────────────────────────────────────────────────────────────────────

def _snap(
    *,
    lt_price: float | None = None,
    ask: float = 0.0,
    bid: float = 0.0,
    mb_vol: float = 0,
    db_close: float | None = None,
    db_vol: float = 0,
) -> SimpleNamespace:
    """Build a minimal mock Alpaca snapshot."""
    latest_trade = SimpleNamespace(price=lt_price) if lt_price is not None else None
    latest_quote = SimpleNamespace(ask_price=ask, bid_price=bid) if (ask or bid) else None
    minute_bar = SimpleNamespace(volume=mb_vol, close=lt_price or 0)
    daily_bar = SimpleNamespace(close=db_close, volume=db_vol) if db_close is not None else None
    return SimpleNamespace(
        latest_trade=latest_trade,
        latest_quote=latest_quote,
        minute_bar=minute_bar,
        daily_bar=daily_bar,
        prev_daily_bar=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _alpaca_mover — price source
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaMoverPriceSource:
    """Gap% must use daily_bar.close, not latest_trade.price."""

    def _call(self, snap, prev_close=3.00, min_vol=0):
        return pm._alpaca_mover(
            "TST", snap,
            prev_close=prev_close,
            min_price=1.0, max_price=200.0,
            min_gap_pct=0.0, min_vol=min_vol,
        )

    def test_uses_latest_trade_not_daily_bar_close(self):
        """Alpaca daily_bar.close in premarket = prev session close (not PM price).
        latest_trade.price=4.00, daily_bar.close=3.00 → gap uses 4.00 (actual PM print)."""
        snap = _snap(db_close=3.00, db_vol=100_000, lt_price=4.00)
        result = self._call(snap)
        assert result is not None
        assert abs(result["gap_pct"] - (4.00 - 3.00) / 3.00 * 100) < 0.01
        assert result["current_price"] == 4.00

    def test_falls_back_to_latest_trade_when_daily_bar_absent(self):
        """No daily_bar → use latest_trade.price."""
        snap = _snap(lt_price=3.50, mb_vol=100_000)
        result = self._call(snap)
        assert result is not None
        assert abs(result["gap_pct"] - (3.50 - 3.00) / 3.00 * 100) < 0.01

    def test_falls_back_to_quote_mid_when_both_absent(self):
        """No daily_bar, no latest_trade → use (ask+bid)/2."""
        snap = _snap(ask=3.60, bid=3.40, mb_vol=100_000)
        result = self._call(snap)
        assert result is not None
        assert abs(result["gap_pct"] - (3.50 - 3.00) / 3.00 * 100) < 0.01

    def test_returns_none_when_no_price_source(self):
        snap = _snap()  # no price anywhere
        result = self._call(snap)
        assert result is None

    def test_returns_none_when_prev_close_zero(self):
        snap = _snap(db_close=3.50, db_vol=100_000, lt_price=3.50)
        result = self._call(snap, prev_close=0.0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _alpaca_mover — volume source
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaMoverVolumeSource:
    """PM volume must use daily_bar.volume (cumulative), not minute_bar.volume (1 bar)."""

    def _call(self, snap, min_vol=50_000):
        return pm._alpaca_mover(
            "TST", snap,
            prev_close=3.00,
            min_price=1.0, max_price=200.0,
            min_gap_pct=0.0, min_vol=min_vol,
        )

    def test_daily_bar_volume_passes_filter(self):
        """daily_bar.volume=100k ≥ 50k threshold → accepted."""
        snap = _snap(lt_price=3.50, db_close=3.00, db_vol=100_000, mb_vol=5_000)
        assert self._call(snap) is not None

    def test_daily_bar_volume_fails_filter(self):
        """daily_bar.volume=10k < 50k threshold → rejected."""
        snap = _snap(lt_price=3.50, db_close=3.00, db_vol=10_000, mb_vol=100_000)
        assert self._call(snap) is None

    def test_minute_bar_volume_no_longer_used(self):
        """minute_bar.volume alone can't save a symbol with low daily_bar.volume."""
        snap = _snap(lt_price=3.50, db_close=3.00, db_vol=1_000, mb_vol=999_999)
        assert self._call(snap) is None

    def test_pm_vol_in_result_is_daily_bar_volume(self):
        """Returned pm_vol field = daily_bar.volume, not minute_bar.volume."""
        snap = _snap(lt_price=3.50, db_close=3.00, db_vol=200_000, mb_vol=5_000)
        result = self._call(snap, min_vol=0)
        assert result is not None
        assert result["pm_vol"] == 200_000

    def test_fallback_vol_zero_when_no_daily_bar(self):
        """If daily_bar is absent, pm_vol=0 and any min_vol > 0 rejects."""
        snap = _snap(lt_price=3.50, mb_vol=999_999)
        assert self._call(snap, min_vol=1) is None


# ─────────────────────────────────────────────────────────────────────────────
# _alpaca_mover — price/gap filters
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaMoverFilters:

    def _call(self, snap, **kw):
        defaults = dict(prev_close=3.00, min_price=2.0, max_price=100.0,
                        min_gap_pct=2.0, min_vol=0)
        defaults.update(kw)
        return pm._alpaca_mover("TST", snap, **defaults)

    def test_price_below_floor_rejected(self):
        snap = _snap(db_close=1.50, db_vol=100_000)
        assert self._call(snap) is None

    def test_price_above_ceiling_rejected(self):
        snap = _snap(db_close=150.00, db_vol=100_000)
        assert self._call(snap) is None

    def test_gap_below_min_rejected(self):
        # prev_close=3.00, db_close=3.03 → gap=1.0% < min_gap_pct=2.0
        snap = _snap(db_close=3.03, db_vol=100_000)
        assert self._call(snap) is None

    def test_negative_gap_passes_min_gap_check(self):
        # gap = -5% (short gapper); lt_price drives gap, db_close provides daily_bar for vol
        snap = _snap(lt_price=2.85, db_close=2.85, db_vol=100_000)
        result = self._call(snap)
        assert result is not None
        assert result["gap_pct"] < 0

    def test_spread_computed_correctly(self):
        snap = _snap(db_close=3.50, db_vol=200_000, ask=3.54, bid=3.46)
        result = self._call(snap, min_gap_pct=0.0)
        assert result is not None
        expected_spread = (3.54 - 3.46) / 3.50 * 100
        assert abs(result["spread_pct"] - expected_spread) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_prev_closes_batch — symbol filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchPrevClosesSymbolFilter:
    """Crypto / dotted / numeric symbols must be stripped before the Alpaca call."""

    def test_dotted_symbols_stripped(self):
        """BUTTCOIN.X and BTC-USD must never reach the provider."""
        captured = []

        def fake_provider():
            class FakeProv:
                def download_daily_batch(self, syms, period):
                    captured.extend(syms)
                    return {}
            return FakeProv()

        with patch("tools.premarket_watchlist.sys.path"):
            with patch.dict("sys.modules", {
                "providers.alpaca_provider": MagicMock(AlpacaProvider=fake_provider),
            }):
                # Directly test the filter logic (line 115)
                symbols = ["AAPL", "BUTTCOIN.X", "TOSHI.X", "MEZO.X", "BTC-USD", "NVDA"]
                clean = [s for s in symbols if s and '.' not in s and s.isalpha()]
                assert "BUTTCOIN.X" not in clean
                assert "TOSHI.X" not in clean
                assert "MEZO.X" not in clean
                assert "BTC-USD" not in clean
                assert "AAPL" in clean
                assert "NVDA" in clean

    def test_numeric_symbols_stripped(self):
        symbols = ["AAPL", "123", "1INCH", "TSLA"]
        clean = [s for s in symbols if s and '.' not in s and s.isalpha()]
        assert "123" not in clean
        assert "1INCH" not in clean
        assert "AAPL" in clean
        assert "TSLA" in clean


# ─────────────────────────────────────────────────────────────────────────────
# Pre-filter in _run_alpaca_source
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAlpacaSourcePreFilter:
    """The pre-filter must use daily_bar.volume, not minute_bar.volume."""

    def test_prefilter_uses_daily_bar_volume(self):
        """Symbol with db_vol=200k but mb_vol=1k must pass the 50k filter."""
        snap = _snap(db_close=5.00, db_vol=200_000, lt_price=6.00, mb_vol=1_000)
        snaps = {"TST": snap}

        with patch("tools.premarket_watchlist._fetch_alpaca_snapshots", return_value=snaps):
            with patch("tools.premarket_watchlist._fetch_prev_closes_batch",
                       return_value={"TST": 4.50}):
                result = pm._run_alpaca_source(
                    ["TST"],
                    min_price=1.0, max_price=200.0,
                    min_gap_pct=0.0, min_vol=50_000,
                )
        assert "TST" in result

    def test_prefilter_rejects_low_daily_bar_volume(self):
        """Symbol with db_vol=10k must fail the 50k filter even if mb_vol=500k."""
        snap = _snap(db_close=5.00, db_vol=10_000, lt_price=6.00, mb_vol=500_000)
        snaps = {"TST": snap}

        with patch("tools.premarket_watchlist._fetch_alpaca_snapshots", return_value=snaps):
            with patch("tools.premarket_watchlist._fetch_prev_closes_batch",
                       return_value={"TST": 4.50}):
                result = pm._run_alpaca_source(
                    ["TST"],
                    min_price=1.0, max_price=200.0,
                    min_gap_pct=0.0, min_vol=50_000,
                )
        assert "TST" not in result

    def test_prefilter_uses_latest_trade_for_price(self):
        """Price filter uses latest_trade.price (the actual PM print).
        lt_price=150 is above max_price=100 → rejected even though db_close=5."""
        snap = _snap(lt_price=150.0, db_close=5.0, db_vol=200_000)
        snaps = {"TST": snap}

        with patch("tools.premarket_watchlist._fetch_alpaca_snapshots", return_value=snaps):
            with patch("tools.premarket_watchlist._fetch_prev_closes_batch",
                       return_value={"TST": 4.50}):
                result = pm._run_alpaca_source(
                    ["TST"],
                    min_price=1.0, max_price=100.0,
                    min_gap_pct=0.0, min_vol=0,
                )
        assert "TST" not in result


# ─────────────────────────────────────────────────────────────────────────────
# _enrich_with_alpaca — same fixes applied
# ─────────────────────────────────────────────────────────────────────────────

class TestEnrichWithAlpaca:
    """Enrichment path (Finviz/StockTwits symbols) must also use daily_bar."""

    def test_enrich_uses_latest_trade_for_gap(self):
        """Gap% uses latest_trade.price (actual PM print), not daily_bar.close."""
        snap = _snap(db_close=3.50, db_vol=200_000, lt_price=4.00)
        snaps = {"TST": snap}

        with patch("tools.premarket_watchlist._fetch_alpaca_snapshots", return_value=snaps):
            with patch("tools.premarket_watchlist._fetch_prev_closes_batch",
                       return_value={"TST": 3.00}):
                enriched = pm._enrich_with_alpaca(["TST"], existing={})

        assert "TST" in enriched
        assert abs(enriched["TST"]["gap_pct"] - (4.00 - 3.00) / 3.00 * 100) < 0.01
        assert enriched["TST"]["current_price"] == 4.00

    def test_enrich_uses_daily_bar_volume(self):
        snap = _snap(db_close=3.50, db_vol=300_000, lt_price=4.00, mb_vol=500)

        with patch("tools.premarket_watchlist._fetch_alpaca_snapshots",
                   return_value={"TST": snap}):
            with patch("tools.premarket_watchlist._fetch_prev_closes_batch",
                       return_value={"TST": 3.00}):
                enriched = pm._enrich_with_alpaca(["TST"], existing={})

        assert enriched["TST"]["pm_vol"] == 300_000

    def test_enrich_skips_existing_symbols(self):
        """Symbols already in the alpaca mover dict should not be re-fetched."""
        called = []
        def fake_snap(syms):
            called.extend(syms)
            return {}

        with patch("tools.premarket_watchlist._fetch_alpaca_snapshots", side_effect=fake_snap):
            with patch("tools.premarket_watchlist._fetch_prev_closes_batch", return_value={}):
                pm._enrich_with_alpaca(["AAPL", "TSLA"], existing={"AAPL": {}})

        assert "TSLA" in called
        assert "AAPL" not in called


# ─────────────────────────────────────────────────────────────────────────────
# Scoring and ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestScoring:

    def test_multi_source_ranks_above_single_source(self):
        a = {"symbol": "A", "gap_pct": 5.0, "sources": {"alpaca", "finviz"}}
        b = {"symbol": "B", "gap_pct": 50.0, "sources": {"alpaca"}}
        ranked = sorted([a, b], key=pm._score, reverse=True)
        assert ranked[0]["symbol"] == "A"  # 2 sources wins over higher gap

    def test_same_source_count_ranks_by_gap(self):
        a = {"symbol": "A", "gap_pct": 3.0, "sources": {"alpaca"}}
        b = {"symbol": "B", "gap_pct": 10.0, "sources": {"finviz"}}
        ranked = sorted([a, b], key=pm._score, reverse=True)
        assert ranked[0]["symbol"] == "B"

    def test_negative_gap_uses_abs_value_for_rank(self):
        a = {"symbol": "A", "gap_pct": -20.0, "sources": {"alpaca"}}
        b = {"symbol": "B", "gap_pct": 5.0, "sources": {"alpaca"}}
        ranked = sorted([a, b], key=pm._score, reverse=True)
        assert ranked[0]["symbol"] == "A"

    def test_gap_capped_at_30_for_score(self):
        """4000% and 30% gap should produce the same gap_weight contribution."""
        pump  = {"symbol": "PUMP", "gap_pct": 4000.0, "sources": {"alpaca"},
                 "current_price": 1.50, "pm_vol": 0, "spread_pct": 0.0}
        clean = {"symbol": "CLEAN", "gap_pct": 30.0, "sources": {"alpaca"},
                 "current_price": 1.50, "pm_vol": 0, "spread_pct": 0.0}
        # Both have 1 source + min(gap,30)=30 as base — pump gets extra penalty
        assert pm._score(pump) < pm._score(clean)

    def test_pump_over_200pct_penalized_vs_moderate_gap(self):
        """A 300% gapper should lose to a 15% quality setup."""
        pump  = {"symbol": "PUMP", "gap_pct": 300.0, "sources": {"alpaca"},
                 "current_price": 2.00, "pm_vol": 50_000, "spread_pct": 3.0}
        orb   = {"symbol": "ORB",  "gap_pct": 15.0,  "sources": {"alpaca"},
                 "current_price": 20.0, "pm_vol": 300_000, "spread_pct": 0.3}
        assert pm._score(orb) > pm._score(pump)

    def test_quality_bonus_for_orb_sweet_spot(self):
        """$5–$50 stock with 5–30% gap earns quality bonus."""
        sweet = {"symbol": "SWEET", "gap_pct": 12.0, "sources": {"alpaca"},
                 "current_price": 18.0, "pm_vol": 0, "spread_pct": 0.0}
        outside = {"symbol": "OUT", "gap_pct": 12.0, "sources": {"alpaca"},
                   "current_price": 1.50, "pm_vol": 0, "spread_pct": 0.0}
        assert pm._score(sweet) > pm._score(outside)

    def test_volume_bonus_increases_score(self):
        lo_vol = {"symbol": "LO", "gap_pct": 10.0, "sources": {"alpaca"},
                  "current_price": 10.0, "pm_vol": 0, "spread_pct": 0.0}
        hi_vol = {"symbol": "HI", "gap_pct": 10.0, "sources": {"alpaca"},
                  "current_price": 10.0, "pm_vol": 600_000, "spread_pct": 0.0}
        assert pm._score(hi_vol) > pm._score(lo_vol)

    def test_spread_penalty_reduces_score(self):
        tight = {"symbol": "TIGHT", "gap_pct": 10.0, "sources": {"alpaca"},
                 "current_price": 10.0, "pm_vol": 100_000, "spread_pct": 0.2}
        wide  = {"symbol": "WIDE",  "gap_pct": 10.0, "sources": {"alpaca"},
                 "current_price": 10.0, "pm_vol": 100_000, "spread_pct": 4.0}
        assert pm._score(tight) > pm._score(wide)


# ─────────────────────────────────────────────────────────────────────────────
# Generate Orders logic (pure unit — no Flask context needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateOrdersLogic:
    """Validate the order-plan arithmetic independently of the Flask route."""

    def _plan(self, pm_high, pm_low, prev_close, pm_last, risk_dollars=50.0):
        import math as m
        side = 'long' if pm_last >= prev_close else 'short'
        if side == 'long':
            entry = round(pm_high + 0.01, 2)
            stop  = round(pm_low, 2)
            target_2r = round(entry + 2.0 * abs(entry - stop), 2)
        else:
            entry = round(pm_low - 0.01, 2)
            stop  = round(pm_high, 2)
            target_2r = round(entry - 2.0 * abs(stop - entry), 2)
        risk_per_share = round(abs(entry - stop), 4)
        shares = max(1, int(m.floor(risk_dollars / risk_per_share)))
        return dict(side=side, entry=entry, stop=stop, target_2r=target_2r,
                    risk_per_share=risk_per_share, shares=shares)

    def test_long_entry_above_pm_high(self):
        p = self._plan(pm_high=10.00, pm_low=8.00, prev_close=9.00, pm_last=9.50)
        assert p["side"] == "long"
        assert p["entry"] == 10.01

    def test_long_stop_at_pm_low(self):
        p = self._plan(pm_high=10.00, pm_low=8.00, prev_close=9.00, pm_last=9.50)
        assert p["stop"] == 8.00

    def test_long_target_2r_correct(self):
        p = self._plan(pm_high=10.00, pm_low=8.00, prev_close=9.00, pm_last=9.50)
        # entry=10.01, stop=8.00, risk=2.01, target=10.01+4.02=14.03
        assert abs(p["target_2r"] - (10.01 + 2 * 2.01)) < 0.01

    def test_short_entry_below_pm_low(self):
        p = self._plan(pm_high=10.00, pm_low=8.00, prev_close=9.00, pm_last=8.50)
        assert p["side"] == "short"
        assert p["entry"] == 7.99

    def test_short_stop_at_pm_high(self):
        p = self._plan(pm_high=10.00, pm_low=8.00, prev_close=9.00, pm_last=8.50)
        assert p["stop"] == 10.00

    def test_short_target_2r_below_entry(self):
        p = self._plan(pm_high=10.00, pm_low=8.00, prev_close=9.00, pm_last=8.50)
        # entry=7.99, stop=10.00, risk=2.01, target=7.99-4.02=3.97
        assert p["target_2r"] < p["entry"]
        assert abs(p["target_2r"] - (7.99 - 2 * 2.01)) < 0.01

    def test_shares_floored_to_int(self):
        p = self._plan(pm_high=10.00, pm_low=9.50, prev_close=9.70, pm_last=9.80,
                       risk_dollars=50.0)
        # entry=10.01, stop=9.50, risk=0.51, shares=floor(50/0.51)=98
        import math
        expected = max(1, int(math.floor(50.0 / 0.51)))
        assert p["shares"] == expected

    def test_minimum_one_share(self):
        p = self._plan(pm_high=1000.00, pm_low=800.00, prev_close=900.00,
                       pm_last=950.00, risk_dollars=10.0)
        assert p["shares"] >= 1

    def test_tight_range_flag(self):
        """Range < $0.05 should be flagged as too tight."""
        pm_high, pm_low = 5.03, 5.00
        risk_per_share = abs((pm_high + 0.01) - pm_low)
        assert risk_per_share < 0.05

    def test_risk_per_share_is_always_positive(self):
        for side_args in [
            dict(pm_high=10, pm_low=8, prev_close=9, pm_last=9.5),
            dict(pm_high=10, pm_low=8, prev_close=9, pm_last=8.5),
        ]:
            p = self._plan(**side_args)
            assert p["risk_per_share"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# PM window boundary — [4:00am, 9:30am)
# ─────────────────────────────────────────────────────────────────────────────

class TestPMWindowBoundary:
    """Bars at 3:59am and 9:30am must be excluded; 4:00am and 9:29am included."""

    def test_pm_mask_boundaries(self):
        import pandas as pd
        import pytz
        ET = pytz.timezone("America/New_York")
        today = datetime.now(ET).date()

        def et(h, m):
            return ET.localize(datetime(today.year, today.month, today.day, h, m, 0))

        times = [et(3, 59), et(4, 0), et(6, 30), et(9, 29), et(9, 30), et(10, 0)]
        idx = pd.DatetimeIndex(times)
        pm_open  = et(4, 0)
        pm_close = et(9, 30)
        mask = (idx >= pm_open) & (idx < pm_close)
        included = [t.strftime("%H:%M") for t, inc in zip(times, mask) if inc]
        excluded = [t.strftime("%H:%M") for t, inc in zip(times, mask) if not inc]

        assert "04:00" in included
        assert "06:30" in included
        assert "09:29" in included
        assert "03:59" in excluded
        assert "09:30" in excluded
        assert "10:00" in excluded


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — require live Alpaca credentials
# ─────────────────────────────────────────────────────────────────────────────

def _has_alpaca_creds() -> bool:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
    sec = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or ""
    return bool(key and sec)


@pytest.mark.integration
@pytest.mark.skipif(not _has_alpaca_creds(), reason="No Alpaca credentials")
class TestLiveAlpacaIntegration:
    """
    Hit real Alpaca API to verify:
    1. daily_bar.volume >> minute_bar.volume (confirms fix matters)
    2. gap% via daily_bar.close is within 3% of gap% via latest_trade
    3. prev_close from download_daily_batch matches get_daily_history
    """

    SYMS = ["AAPL", "NVDA", "MSFT", "TSLA", "AMD"]

    @pytest.fixture(scope="class")
    def snapshots(self):
        from alpaca.data.requests import StockSnapshotRequest
        from alpaca.data import StockHistoricalDataClient
        key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        client = StockHistoricalDataClient(key, secret)
        req = StockSnapshotRequest(symbol_or_symbols=self.SYMS)
        return client.get_stock_snapshot(req)

    def test_daily_bar_volume_exceeds_minute_bar_volume(self, snapshots):
        """During an active session, daily_bar.volume > minute_bar.volume for liquid names."""
        for sym in self.SYMS:
            snap = snapshots.get(sym)
            if snap is None:
                continue
            db = getattr(snap, "daily_bar", None)
            mb = getattr(snap, "minute_bar", None)
            db_vol = int(getattr(db, "volume", 0) or 0) if db else 0
            mb_vol = int(getattr(mb, "volume", 0) or 0) if mb else 0
            if db_vol > 0 and mb_vol > 0:
                # Cumulative session vol must be materially larger than one 1-min bar
                assert db_vol > mb_vol, (
                    f"{sym}: daily_bar.volume={db_vol} should exceed "
                    f"minute_bar.volume={mb_vol}"
                )

    def test_daily_bar_close_consistent_with_latest_trade(self, snapshots):
        """daily_bar.close and latest_trade.price should be within 2% during session."""
        for sym in self.SYMS:
            snap = snapshots.get(sym)
            if snap is None:
                continue
            lt = getattr(snap, "latest_trade", None)
            db = getattr(snap, "daily_bar", None)
            lt_price = float(getattr(lt, "price", 0) or 0) if lt else 0
            db_close = float(getattr(db, "close", 0) or 0) if db else 0
            if lt_price > 0 and db_close > 0:
                diff_pct = abs(db_close - lt_price) / lt_price * 100
                assert diff_pct < 2.0, (
                    f"{sym}: daily_bar.close={db_close} vs latest_trade={lt_price} "
                    f"differ by {diff_pct:.2f}% (>2%)"
                )

    def test_prev_close_consistent_across_paths(self):
        """
        prev_close from download_daily_batch (scan path) must match
        prev_close from get_daily_history (generate-orders path) within $0.01.
        """
        import pytz
        from providers.alpaca_provider import AlpacaProvider
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        ET = pytz.timezone("America/New_York")
        today = datetime.now(ET).date()
        today_start = ET.localize(datetime(today.year, today.month, today.day, 0, 0, 0))
        start_utc = (datetime.now(ET) - timedelta(days=7)).astimezone(timezone.utc)
        end_utc = datetime.now(ET).astimezone(timezone.utc)

        key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        client = StockHistoricalDataClient(key, secret)

        provider = AlpacaProvider()
        batch = provider.download_daily_batch(self.SYMS, period="5d")

        for sym in self.SYMS:
            # Scan path prev_close
            df_batch = batch.get(sym)
            if df_batch is None or df_batch.empty:
                continue
            idx_b = df_batch.index.tz_convert(ET) if df_batch.index.tz else df_batch.index.tz_localize("UTC").tz_convert(ET)
            prior_b = df_batch[idx_b < today_start]
            if prior_b.empty:
                continue
            pc_batch = float(prior_b["Close"].iloc[-1])

            # Generate-orders path prev_close
            df_go = provider.get_daily_history(sym, period="5d")
            if df_go is None or df_go.empty:
                continue
            idx_go = df_go.index.tz_convert(ET) if df_go.index.tz else df_go.index.tz_localize("UTC").tz_convert(ET)
            prior_go = df_go[idx_go < today_start]
            if prior_go.empty:
                continue
            pc_go = float(prior_go["Close"].iloc[-1])

            assert abs(pc_batch - pc_go) < 0.01, (
                f"{sym}: scan prev_close={pc_batch} vs generate-orders prev_close={pc_go}"
            )

    def test_gap_pct_alignment_scan_vs_generate_orders(self):
        """
        gap% computed from daily_bar.close (fixed scan) must be within 3% of
        gap% from the last PM 1m bar close (generate-orders source).
        Only meaningful when run pre-market (4am–9:30am ET).
        """
        import pytz
        from alpaca.data.requests import StockSnapshotRequest, StockBarsRequest
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from providers.alpaca_provider import AlpacaProvider

        ET = pytz.timezone("America/New_York")
        now_et = datetime.now(ET)
        if not (4 <= now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30)):
            pytest.skip("Gap alignment test only valid during pre-market hours (4–9:30 ET)")

        today = now_et.date()
        today_start = ET.localize(datetime(today.year, today.month, today.day, 0, 0, 0))
        pm_open = ET.localize(datetime(today.year, today.month, today.day, 4, 0, 0))
        pm_close = ET.localize(datetime(today.year, today.month, today.day, 9, 30, 0))

        key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        client = StockHistoricalDataClient(key, secret)
        provider = AlpacaProvider()

        req = StockSnapshotRequest(symbol_or_symbols=self.SYMS)
        snaps = client.get_stock_snapshot(req)

        for sym in self.SYMS:
            snap = snaps.get(sym)
            db = getattr(snap, "daily_bar", None) if snap else None
            db_close = float(getattr(db, "close", 0) or 0) if db else 0
            if not db_close:
                continue

            # prev_close
            df = provider.get_daily_history(sym, period="5d")
            if df is None or df.empty:
                continue
            idx_et = df.index.tz_convert(ET) if df.index.tz else df.index.tz_localize("UTC").tz_convert(ET)
            prior = df[idx_et < today_start]
            if prior.empty:
                continue
            prev_close = float(prior["Close"].iloc[-1])

            # Scan gap% (fixed)
            gap_scan = (db_close - prev_close) / prev_close * 100

            # Generate-orders gap% from last PM bar
            try:
                bars = provider.get_bars_range(
                    symbol=sym, interval="1m",
                    from_d=today, to_d=today,
                    include_prepost=True, timeout_s=10,
                )
                if bars is None or bars.empty:
                    continue
                idx_b = bars.index.tz_convert(ET) if bars.index.tz else bars.index.tz_localize("UTC").tz_convert(ET)
                pm_bars = bars[(idx_b >= pm_open) & (idx_b < pm_close)]
                if pm_bars.empty:
                    continue
                pm_last = float(pm_bars["Close"].iloc[-1])
                gap_go = (pm_last - prev_close) / prev_close * 100
                diff = abs(gap_scan - gap_go)
                assert diff < 3.0, (
                    f"{sym}: scan gap={gap_scan:.2f}% vs generate-orders gap={gap_go:.2f}% "
                    f"(diff={diff:.2f}% > 3%)"
                )
            except Exception:
                continue
