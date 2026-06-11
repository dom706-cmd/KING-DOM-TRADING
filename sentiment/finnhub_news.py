from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class FinnhubNewsSentiment:
    """Real Finnhub-backed sentiment/news fetcher with FinBERT re-scoring.

    Tenant rules:
    - No fake data. If Finnhub fails, raise a real exception.
    - Headlines fetched from /company-news, scored with FinBERT.
    - Falls back to Finnhub's own aggregate score only if no headlines returned.
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

    def fetch_company_news(self, symbol: str, days_back: int = 3, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent company news headlines from Finnhub /company-news."""
        now = datetime.now(timezone.utc)
        date_from = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = now.strftime("%Y-%m-%d")
        data = self._get("/company-news", {
            "symbol": symbol.strip().upper(),
            "from": date_from,
            "to": date_to,
        })
        if not isinstance(data, list):
            raise RuntimeError(f"Finnhub /company-news returned unexpected type for {symbol}: {type(data)}")
        articles = []
        for a in data[:limit]:
            if not isinstance(a, dict):
                continue
            headline = (a.get("headline") or a.get("summary") or "").strip()
            if headline:
                articles.append({
                    "title": headline,
                    "url": a.get("url") or "",
                    "published_at": str(a.get("datetime") or ""),
                    "source": a.get("source") or "",
                })
        return articles

    def news_sentiment(self, symbol: str, limit: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
        """Return (score, headlines).

        Fetches real headlines via /company-news and scores with FinBERT.
        If no headlines returned, uses Finnhub aggregate score from /news-sentiment.
        Raises if no usable data at all.
        """
        from sentiment.finbert_sentiment import score_headlines

        symbol = symbol.strip().upper()

        # Primary path: fetch real headlines, score with FinBERT
        try:
            articles = self.fetch_company_news(symbol, days_back=3, limit=limit)
        except Exception as e:
            raise RuntimeError(f"Finnhub /company-news failed for {symbol}: {e}") from e

        if articles:
            titles = [a["title"] for a in articles if a.get("title")]
            if not titles:
                raise RuntimeError(f"Finnhub returned articles but no headline text for {symbol}")
            score = score_headlines(titles)
            return float(score), articles

        # Secondary path: no articles in last 3 days — use aggregate score from /news-sentiment
        payload = self._get("/news-sentiment", {"symbol": symbol})
        agg_score: Optional[float] = None
        if isinstance(payload, dict):
            sent = payload.get("sentiment")
            if isinstance(sent, dict):
                s = sent.get("score")
                if isinstance(s, (int, float)):
                    agg_score = float(s)
            if agg_score is None:
                s2 = payload.get("companyNewsScore")
                if isinstance(s2, (int, float)):
                    agg_score = float(s2)
            if agg_score is None:
                bull = payload.get("bullishPercent")
                bear = payload.get("bearishPercent")
                if isinstance(bull, (int, float)) and isinstance(bear, (int, float)):
                    agg_score = float(bull) - float(bear)

        if agg_score is None:
            raise RuntimeError(f"Finnhub returned no headlines and no aggregate score for {symbol}")

        return float(agg_score), []
