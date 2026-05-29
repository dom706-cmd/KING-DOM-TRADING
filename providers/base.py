from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import pandas as pd

@dataclass(frozen=True)
class BarsRequest:
    symbol: str
    interval: str  # "1m", "5m", "1d"
    period: str    # e.g. "1d", "5d", "1mo"
    include_prepost: bool = False

class MarketDataProvider(Protocol):
    name: str
    def get_bars(self, req: BarsRequest, timeout_s: int | None = None) -> pd.DataFrame: ...
    def get_daily_history(self, symbol: str, period: str = "6mo", timeout_s: int | None = None) -> pd.DataFrame: ...
