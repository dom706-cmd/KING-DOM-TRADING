from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class FinnhubNewsSentiment:
    """Real Finnhub-backed sentiment/news fetcher.

    Tenant rules:
    - No fake data. If Finnhub fails, raise a real exception.
    - Return real headlines + a real score derived from Finnhub payload.
    """

    api_key: str
    base_url: str = "https://finnhub.io/api/v1"
    timeout_sec: int = 15

    @classmethod
    def from_env(cls) -> "FinnhubNewsSentiment":
        key = (os.getenv("FINNHUB_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("FINNHUB_API_KEY is not set")
        return cls(api_key=key)

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        params = dict(params)
        params["token"] = self.api_key
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params, timeout=self.timeout_sec)
        if r.status_code != 200:
            raise RuntimeError(f"Finnhub HTTP {r.status_code} for {path}: {r.text[:300]}")
        return r.json()

    def news_sentiment(self, symbol: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Return (score, headlines).

        Score is derived from Finnhub /news-sentiment response.
        If Finnhub returns no data, raise a real exception (no placeholders).
        """
        symbol = symbol.strip().upper()
        payload = self._get("/news-sentiment", {"symbol": symbol})

        # Finnhub returns various metrics; we use the overall sentiment score if present,
        # else fall back to aggregating bullish/bearish from the payload.
        score: Optional[float] = None
        if isinstance(payload, dict):
            # common key seen in docs is 'sentiment' with 'score' or similar; handle safely.
            sent = payload.get("sentiment")
            if isinstance(sent, dict):
                s = sent.get("score")
                if isinstance(s, (int, float)):
                    score = float(s)

            # some payloads include 'buzz' and 'companyNewsScore' / 'sectorAverageBullishPercent' etc.
            if score is None:
                s2 = payload.get("companyNewsScore")
                if isinstance(s2, (int, float)):
                    score = float(s2)

        articles = payload.get("articles") if isinstance(payload, dict) else None
        headlines: List[Dict[str, Any]] = []
        if isinstance(articles, list):
            for a in articles[:20]:
                if isinstance(a, dict):
                    headlines.append(a)

        if score is None:
            # As a conservative fallback, compute a simple score from bullish/bearish percentages if present.
            # If nothing usable exists, raise.
            bull = payload.get("bullishPercent") if isinstance(payload, dict) else None
            bear = payload.get("bearishPercent") if isinstance(payload, dict) else None
            if isinstance(bull, (int, float)) and isinstance(bear, (int, float)):
                score = float(bull) - float(bear)

        if score is None:
            raise RuntimeError("Finnhub returned no usable sentiment score for symbol")

        return float(score), headlines
