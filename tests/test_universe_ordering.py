from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from universe.nasdaq_symbols import UniverseConfig, _rank_symbols, fetch_us_equity_symbols


class UniverseOrderingTests(unittest.TestCase):
    def test_rank_symbols_prioritizes_tradability_quality_over_alphabetical_order(self) -> None:
        ordered = _rank_symbols(
            ["ZZZZ", "AAAA", "QQQQQ"],
            listing_meta={
                "ZZZZ": {"market_category": "Q", "financial_status": "N", "etf": "N"},
                "AAAA": {"market_category": "S", "financial_status": "N", "etf": "N"},
                "QQQQQ": {"market_category": "Q", "financial_status": "N", "etf": "N"},
            },
            asset_meta={
                "ZZZZ": {
                    "exchange": "NASDAQ",
                    "tradable": True,
                    "easy_to_borrow": True,
                    "shortable": True,
                    "fractionable": True,
                    "maintenance_margin_requirement": 30,
                    "attributes": ["has_options", "fractional_eh_enabled"],
                },
                "AAAA": {
                    "exchange": "NASDAQ",
                    "tradable": True,
                    "easy_to_borrow": False,
                    "shortable": False,
                    "fractionable": False,
                    "maintenance_margin_requirement": 100,
                    "attributes": [],
                },
                "QQQQQ": {
                    "exchange": "NASDAQ",
                    "tradable": True,
                    "easy_to_borrow": True,
                    "shortable": True,
                    "fractionable": True,
                    "maintenance_margin_requirement": 30,
                    "attributes": ["has_options"],
                },
            },
        )
        self.assertEqual(ordered[0], "ZZZZ")
        self.assertEqual(ordered[-1], "AAAA")

    def test_fetch_us_equity_symbols_falls_back_to_cached_universe_when_listing_fetch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "us_symbols.txt"
            cache_file.write_text("NVDA\nAAPL\nAMD\n", encoding="utf-8")
            with patch("universe.nasdaq_symbols._fetch_text", side_effect=RuntimeError("dns down")):
                ordered = fetch_us_equity_symbols(UniverseConfig(cache_dir=tmpdir))
        self.assertEqual(ordered[:3], ["NVDA", "AAPL", "AMD"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
