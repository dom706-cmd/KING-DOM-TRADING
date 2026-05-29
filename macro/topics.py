from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import re


@dataclass(frozen=True)
class TopicRule:
    key: str
    label: str
    keywords: List[str]
    sectors: List[str]  # yfinance sector strings


TOPIC_RULES: List[TopicRule] = [
    TopicRule("energy", "Energy & Oil/Gas", ["oil", "gas", "energy", "lng", "opec", "pipeline", "refinery", "petroleum"], ["Energy"]),
    TopicRule("defense", "Defense & Geopolitics", ["defense", "pentagon", "nato", "missile", "military", "war", "conflict", "ukraine", "israel", "taiwan"], ["Industrials"]),
    TopicRule("semis", "Semiconductors & Chips", ["semiconductor", "chip", "chips", "tsmc", "export control", "ai chip"], ["Technology"]),
    TopicRule("banking", "Banking & Credit", ["bank", "banks", "credit", "fdic", "rates", "inflation", "treasury yield"], ["Financial Services"]),
    TopicRule("health", "Healthcare & Pharma", ["fda", "drug", "pharma", "health", "medicare", "vaccin"], ["Healthcare"]),
    TopicRule("trade", "Trade, Tariffs & Sanctions", ["tariff", "sanction", "export", "import", "trade", "customs", "embargo"], ["Industrials", "Technology", "Consumer Cyclical"]),
    TopicRule("energy_transition", "Energy Transition", ["solar", "wind", "battery", "ev", "renewable", "grid"], ["Utilities", "Technology", "Consumer Cyclical"]),
    TopicRule("doj_enforcement", "DOJ Enforcement", ["doj", "antitrust", "indict", "settlement", "fraud", "investigation"], ["Communication Services", "Technology", "Financial Services"]),
]

_WORD = re.compile(r"[a-z0-9']+")


def _norm(text: str) -> str:
    return (text or "").lower()


def classify_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Attach topic hits to each item and produce counts."""
    counts = {r.key: 0 for r in TOPIC_RULES}
    tagged: List[Dict[str, Any]] = []

    for it in items:
        text = _norm((it.get("title") or "") + " " + (it.get("summary") or ""))
        hits = []
        for rule in TOPIC_RULES:
            for kw in rule.keywords:
                if kw in text:
                    hits.append(rule.key)
                    break
        for h in set(hits):
            counts[h] += 1
        it2 = dict(it)
        it2["topics"] = hits
        tagged.append(it2)

    return {"items": tagged, "topic_counts": counts}


def score_relevance_for_symbol(items: List[Dict[str, Any]], symbol_sector: Optional[str]) -> List[Dict[str, Any]]:
    """Return items sorted by simple relevance to a symbol via sector mapping + keyword hits."""
    sec = (symbol_sector or "").strip()
    def rel(it: Dict[str, Any]) -> float:
        topics = it.get("topics") or []
        base = float(len(topics))
        if not sec:
            return base
        for rule in TOPIC_RULES:
            if rule.key in topics and sec in rule.sectors:
                base += 2.0
        return base

    out = list(items)
    out.sort(key=rel, reverse=True)
    return out
