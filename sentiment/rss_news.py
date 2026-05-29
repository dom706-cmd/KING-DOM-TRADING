
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xml.etree import ElementTree as ET


@dataclass
class RSSItem:
    title: str
    link: str
    published: str | None


class RSSNewsSentiment:
    """Free sentiment via Google News RSS + VADER.

    Tenants:
      - Real scrape or real failure (raises on HTTP errors)
      - No fake data
    """

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.analyzer = SentimentIntensityAnalyzer()

    @staticmethod
    def _rss_url(symbol: str) -> str:
        q = urllib.parse.quote_plus(f"{symbol} stock")
        return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    def fetch_headlines(self, symbol: str, limit: int = 10, timeout_s: int = 10) -> List[Dict[str, Any]]:
        url = self._rss_url(symbol)
        r = self.session.get(url, timeout=timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"Google News RSS HTTP {r.status_code} for {symbol}")
        try:
            root = ET.fromstring(r.text)
        except Exception as e:
            raise RuntimeError(f"RSS parse failed for {symbol}: {e}") from e
        items = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip() or None
            if title:
                items.append({"title": title, "url": link, "published_at": pub})
            if len(items) >= limit:
                break
        return items

    def score(self, symbol: str, limit: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
        items = self.fetch_headlines(symbol, limit=limit)
        if not items:
            # No news isn't fake; treat as neutral with empty headlines.
            return 0.0, []
        scores = []
        for it in items:
            s = self.analyzer.polarity_scores(it["title"])["compound"]
            scores.append(float(s))
        return float(sum(scores) / max(1, len(scores))), items
