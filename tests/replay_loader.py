from __future__ import annotations

import json
from datetime import datetime as real_datetime
from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

from providers.base import BarsRequest

ET = ZoneInfo("America/New_York")
REPLAY_DIR = Path(__file__).resolve().parent / "fixtures" / "replay"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "fixtures" / "snapshots"


class ReplayProvider:
    name = "replay"

    def __init__(
        self,
        *,
        daily: dict[str, pd.DataFrame],
        intraday: dict[str, pd.DataFrame],
        latest_trade: dict[str, dict[str, Any]],
        latest_quote: dict[str, dict[str, Any]],
    ) -> None:
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
        from_d,
        to_d,
        include_prepost: bool = False,
        timeout_s: int | None = None,
    ) -> pd.DataFrame:
        return self._intraday.get(str(symbol).upper(), pd.DataFrame()).copy()

    def get_bars(self, req: BarsRequest, timeout_s: int | None = None) -> pd.DataFrame:
        return self._intraday.get(str(req.symbol).upper(), pd.DataFrame()).copy()

    def get_latest_trade(self, symbol: str) -> dict[str, Any]:
        return dict(self._latest_trade[str(symbol).upper()])

    def get_latest_trade_price(self, symbol: str) -> float:
        return float(self._latest_trade[str(symbol).upper()]["price"])

    def get_latest_quote(self, symbol: str) -> dict[str, Any]:
        return dict(self._latest_quote[str(symbol).upper()])

    def get_news(self, *args, **kwargs) -> list[dict[str, Any]]:
        return []


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    ts_values = [r["ts"] for r in rows]
    idx = pd.to_datetime(ts_values, utc=True)
    try:
        idx = idx.tz_convert(ET)
    except Exception:
        pass

    body = []
    for row in rows:
        item = dict(row)
        item.pop("ts", None)
        body.append(item)

    df = pd.DataFrame(body, index=idx)
    return df


def load_replay_fixture(name: str) -> dict[str, Any]:
    return json.loads((REPLAY_DIR / name).read_text())


def build_provider_from_fixture(name: str) -> tuple[ReplayProvider, dict[str, Any]]:
    fixture = load_replay_fixture(name)
    daily = {sym.upper(): _rows_to_frame(rows) for sym, rows in dict(fixture.get("daily") or {}).items()}
    intraday = {sym.upper(): _rows_to_frame(rows) for sym, rows in dict(fixture.get("intraday") or {}).items()}
    latest_trade = {sym.upper(): dict(v) for sym, v in dict(fixture.get("latest_trade") or {}).items()}
    latest_quote = {sym.upper(): dict(v) for sym, v in dict(fixture.get("latest_quote") or {}).items()}
    return ReplayProvider(daily=daily, intraday=intraday, latest_trade=latest_trade, latest_quote=latest_quote), fixture


def frozen_datetime_from_fixture(fixture: dict[str, Any]):
    frozen = pd.Timestamp(fixture["meta"]["frozen_time"], tz=ET).to_pydatetime()

    class FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return frozen

        @classmethod
        def combine(cls, *args, **kwargs):
            return real_datetime.combine(*args, **kwargs)

    return FrozenDateTime


def monitor_now_from_fixture(fixture: dict[str, Any]) -> float:
    return pd.Timestamp(fixture["meta"]["monitor_now"], tz=ET).timestamp()


def fixture_symbols(fixture: dict[str, Any]) -> list[str]:
    syms = fixture.get("symbols")
    if isinstance(syms, list) and syms:
        return [str(x).upper() for x in syms]
    return ["ABCD", "THIN"]


def load_saved_snapshot(name: str) -> dict[str, Any]:
    return json.loads((SNAPSHOT_DIR / name).read_text())
