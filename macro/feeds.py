from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import time
import requests
import feedparser


@dataclass(frozen=True)
class FeedSource:
    key: str
    name: str
    url: str


DEFAULT_SOURCES: List[FeedSource] = [
    FeedSource("whitehouse-briefing-room", "White House — Briefing Room", "https://www.whitehouse.gov/briefing-room/feed/"),
    FeedSource("doj-opa", "DOJ — Office of Public Affairs", "https://www.justice.gov/opa/pressroom/rss.xml"),
    FeedSource("state-press", "State Dept — Press Releases", "https://www.state.gov/press-releases/feed/"),
    FeedSource("treasury-press", "U.S. Treasury — Press Releases", "https://home.treasury.gov/news/press-releases/feed"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MacroFeedClient:
    """Fetch and cache official RSS feeds.

    Rules:
    - Real fetch or real failure: if a feed request fails, raise RuntimeError.
    - Cache to disk to reduce hammering sources.
    """

    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = int(ttl_seconds)

    def fetch_all(self, limit_per_source: int = 10, sources: Optional[List[FeedSource]] = None) -> Dict[str, Any]:
        sources = sources or DEFAULT_SOURCES
        out: Dict[str, Any] = {
            "ok": True,
            "generated_at": _utc_now_iso(),
            "sources": [],
            "items": [],
        }
        all_items: List[Dict[str, Any]] = []
        for src in sources:
            items = self.fetch_source(src, limit=limit_per_source)
            out["sources"].append({"key": src.key, "name": src.name, "url": src.url, "count": len(items)})
            for it in items:
                it["source_key"] = src.key
                it["source_name"] = src.name
                all_items.append(it)

        # Sort newest-first when possible
        def sort_key(x: Dict[str, Any]):
            return x.get("published_ts") or 0

        all_items.sort(key=sort_key, reverse=True)
        out["items"] = all_items
        return out

    def fetch_source(self, source: FeedSource, limit: int = 10) -> List[Dict[str, Any]]:
        cache_path = self.cache_dir / f"macro_{source.key}.json"
        now = time.time()

        # GOLD rule: never read cached feed data as a runtime fallback.
        # We only persist successful live fetches for audit/debug purposes.

        # Real request
        try:
            r = requests.get(source.url, timeout=20, headers={"User-Agent": "orb-scanner-macro/1.0"})
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            parsed = feedparser.parse(r.text)
            if getattr(parsed, "bozo", False) and getattr(parsed, "bozo_exception", None):
                # still may have entries; don't fail solely on bozo if entries exist
                pass
            entries = parsed.entries or []
            if not entries:
                raise RuntimeError("No entries in feed")

            items: List[Dict[str, Any]] = []
            for e in entries[:limit]:
                title = (getattr(e, "title", "") or "").strip()
                link = (getattr(e, "link", "") or "").strip()
                summary = (getattr(e, "summary", "") or "").strip()
                published = (getattr(e, "published", "") or getattr(e, "updated", "") or "").strip()
                published_ts = None
                # feedparser often has published_parsed/updated_parsed
                dt_struct = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
                if dt_struct:
                    published_ts = int(time.mktime(dt_struct))

                items.append({
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": published,
                    "published_ts": published_ts,
                })

            cache_path.write_text(json.dumps({"fetched_at_ts": now, "items": items}, indent=2), encoding="utf-8")
            return items
        except Exception as e:
            raise RuntimeError(f"MacroFeedClient.fetch_source failed for {source.key}: {e}") from e
