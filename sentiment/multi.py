
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from sentiment.finnhub_news import FinnhubNewsSentiment
from sentiment.rss_news import RSSNewsSentiment


@dataclass
class SentimentBundle:
    symbol: str
    score: float
    provider: str
    headlines: List[Dict[str, Any]]


class SentimentService:
    """Sentiment service with real provider(s) only.

    Provider order:
      1) Finnhub (if FINNHUB_API_KEY present) - real API
      2) Google News RSS + VADER (free) - real scraping

    Tenants:
      - No fake data.
      - Real provider or real failure.
      - Empty news => neutral score with empty headlines (not fake).
    """

    def __init__(self, provider: str = "auto"):
        self.provider = (provider or "auto").strip().lower()
        self._fh = None
        self._rss = None

        if self.provider in ("finnhub", "auto"):
            try:
                self._fh = FinnhubNewsSentiment.from_env()
            except Exception:
                self._fh = None

        if self.provider in ("rss", "auto"):
            self._rss = RSSNewsSentiment()

        if self.provider == "finnhub" and self._fh is None:
            raise RuntimeError("FINNHUB_API_KEY not set; cannot use finnhub sentiment")
        if self.provider == "rss" and self._rss is None:
            raise RuntimeError("RSS sentiment not available")

    def fetch(self, symbol: str, limit: int = 10) -> SentimentBundle:
        sym = (symbol or "").strip().upper()
        if not sym:
            raise ValueError("symbol required")
        if self._fh is not None:
            score, headlines = self._fh.news_sentiment(sym, limit=limit)
            return SentimentBundle(symbol=sym, score=float(score), provider="finnhub", headlines=headlines)
        if self._rss is not None:
            score, headlines = self._rss.score(sym, limit=limit)
            return SentimentBundle(symbol=sym, score=float(score), provider="rss", headlines=headlines)
        raise RuntimeError("No sentiment providers available")

    def get(self, symbol: str) -> SentimentBundle:
        return self.fetch(symbol, limit=10)

    def score_symbols(self, symbols: list, limit: int = 5) -> dict:
        """Return {SYMBOL: {"score": float, "provider": str}, ...} for each symbol.
        Symbols that fail or have no news return a neutral score of 0.0.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        result: dict = {}
        syms = [s.strip().upper() for s in (symbols or []) if s and s.strip()]
        if not syms:
            return result

        def _score_one(sym: str):
            try:
                bundle = self.fetch(sym, limit=limit)
                return sym, {"score": float(bundle.score), "provider": bundle.provider}
            except Exception:
                return sym, {"score": None, "provider": "error"}

        workers = min(len(syms), 12)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_score_one, s): s for s in syms}
            for fut in as_completed(futs):
                try:
                    sym, rec = fut.result()
                    result[sym] = rec
                except Exception:
                    pass
        return result
