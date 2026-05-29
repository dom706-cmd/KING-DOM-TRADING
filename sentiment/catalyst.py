from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List

from providers.symbols import to_provider_symbol

POSITIVE_KEYWORDS = {
    'beat': 0.8,
    'beats': 0.9,
    'raised guidance': 1.2,
    'guidance raised': 1.2,
    'guidance increase': 1.0,
    'upgrade': 0.9,
    'upgraded': 1.0,
    'initiated buy': 0.8,
    'price target raised': 0.8,
    'contract': 0.8,
    'award': 0.7,
    'partnership': 0.8,
    'collaboration': 0.7,
    'approval': 1.2,
    'fda': 0.8,
    'breakthrough': 0.7,
    'acquisition': 0.7,
    'acquire': 0.6,
    'buyback': 0.9,
    'repurchase': 0.8,
    'dividend increase': 0.8,
    'record revenue': 1.0,
    'record sales': 1.0,
    'profit': 0.5,
    'strong demand': 0.7,
    'expands': 0.5,
    'launches': 0.4,
}

NEGATIVE_KEYWORDS = {
    'miss': -0.9,
    'misses': -1.0,
    'guidance cut': -1.2,
    'cuts guidance': -1.2,
    'downgrade': -0.9,
    'downgraded': -1.0,
    'price target cut': -0.8,
    'offering': -1.2,
    'public offering': -1.4,
    'dilution': -1.4,
    'dilutive': -1.4,
    'lawsuit': -0.7,
    'sec': -0.4,
    'investigation': -1.0,
    'fraud': -1.5,
    'bankruptcy': -2.0,
    'chapter 11': -2.0,
    'delisting': -1.8,
    'reverse split': -1.2,
    'going concern': -1.4,
    'restatement': -1.1,
    'default': -1.5,
    'warning': -0.5,
    'weak demand': -0.8,
    'halted': -0.8,
    'suspends': -0.8,
}

CATALYST_TAGS = [
    ('earnings', ['earnings', 'eps', 'quarter', 'q1', 'q2', 'q3', 'q4']),
    ('guidance', ['guidance', 'outlook']),
    ('analyst', ['upgrade', 'downgrade', 'price target', 'initiated']),
    ('deal', ['contract', 'award', 'partnership', 'collaboration', 'acquisition', 'merger']),
    ('fda_bio', ['fda', 'approval', 'phase', 'clinical', 'trial']),
    ('capital_raise', ['offering', 'dilution', 'convertible', 'atm']),
    ('legal_risk', ['lawsuit', 'investigation', 'sec', 'fraud']),
    ('bankruptcy_delisting', ['bankruptcy', 'chapter 11', 'delisting']),
    ('corporate_action', ['buyback', 'repurchase', 'dividend', 'reverse split']),
]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _is_news_eligible_symbol(symbol: str) -> bool:
    """
    Alpaca news endpoint is unreliable / rejects many non-common-equity symbols
    such as preferreds, warrants, rights, units, and dotted share classes.

    Keep catalyst/news requests limited to plain common-equity tickers.
    """
    s = to_provider_symbol(symbol)
    if not s:
        return False

    # Reject obvious non-common-equity / structured symbols after provider normalization
    if "." in s:
        return False

    # Common suffixes that often represent units / rights / warrants / prefs
    bad_suffixes = (
        "W",   # warrants (rough filter when presented as ticker suffix)
        "WS",
        "WT",
        "U",   # units
        "R",   # rights
        "P",   # preferred style suffix in some feeds
    )

    # Keep this conservative: only reject suffix pattern if ticker is longer than 4 chars
    # so normal common tickers like "AR", "WU", etc. are not excluded.
    if len(s) >= 5 and s.endswith(bad_suffixes):
        return False

    return True

def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _hours_since(dt: datetime | None, now: datetime) -> float | None:
    if dt is None:
        return None
    return max(0.0, (now - dt).total_seconds() / 3600.0)

def _recency_weight(hours_old: float | None) -> float:
    # Momentum/intraday recency: articles decay fast — a 6h-old article is largely priced in
    if hours_old is None:
        return 0.25
    if hours_old <= 0.5:
        return 1.0   # breaking now
    if hours_old <= 2:
        return 0.85  # fresh, still actionable
    if hours_old <= 6:
        return 0.60  # partially stale — market has had time to react
    if hours_old <= 24:
        return 0.35  # day-old news rarely drives intraday momentum
    if hours_old <= 72:
        return 0.15  # background context only
    return 0.05      # effectively irrelevant for momentum

def _keyword_score(text: str) -> float:
    t = (text or '').lower()
    score = 0.0
    for k, w in POSITIVE_KEYWORDS.items():
        if k in t:
            score += w
    for k, w in NEGATIVE_KEYWORDS.items():
        if k in t:
            score += w
    # dampen runaway repeated-keyword scoring
    return max(-3.0, min(3.0, score))

def _tags_for_text(text: str) -> list[str]:
    t = (text or '').lower()
    out: list[str] = []
    for tag, words in CATALYST_TAGS:
        if any(w in t for w in words):
            out.append(tag)
    return out

@dataclass
class CatalystArticle:
    symbol: str
    headline: str
    summary: str | None
    source: str | None
    created_at: str | None
    url: str | None
    score: float
    recency_weight: float
    tags: list[str]

@dataclass
class CatalystBundle:
    symbol: str
    score: float  # directional [-1,1]
    confidence: float  # [0,100]
    strength: float  # [0,1]
    article_count: int
    source_count: int
    freshness_hours: float | None
    tags: list[str]
    top_headlines: list[str]
    articles: list[CatalystArticle]
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

class CatalystService:
    """News/catalyst scoring using a provider with real news data (e.g. Alpaca REST news).

    No placeholders: returns real scores when news is available, or records real error/empty state.
    """

    def __init__(self, provider: Any):
        self.provider = provider

    def fetch_batch(
        self,
        symbols: Iterable[str],
        *,
        per_symbol_limit: int = 6,
        lookback_hours: int = 72,
    ) -> dict[str, CatalystBundle]:
        syms = [str(s or '').strip().upper() for s in symbols if str(s or '').strip()]
        if not syms:
            return {}

        if not hasattr(self.provider, 'get_news_batch'):
            return {
                s: CatalystBundle(
                    symbol=s,
                    score=0.0,
                    confidence=0.0,
                    strength=0.0,
                    article_count=0,
                    source_count=0,
                    freshness_hours=None,
                    tags=[],
                    top_headlines=[],
                    articles=[],
                    error='provider_missing_get_news_batch',
                )
                for s in syms
            }

        now = _now_utc()
        start = (now - timedelta(hours=max(1, int(lookback_hours)))).isoformat()

        eligible_syms = [s for s in syms if _is_news_eligible_symbol(s)]
        ineligible_syms = [s for s in syms if s not in eligible_syms]

        def _empty_bundle(sym: str, *, error: str | None = None) -> CatalystBundle:
            return CatalystBundle(
                symbol=sym,
                score=0.0,
                confidence=0.0,
                strength=0.0,
                article_count=0,
                source_count=0,
                freshness_hours=None,
                tags=[],
                top_headlines=[],
                articles=[],
                error=error,
            )

        def _fetch_news_resilient(batch: list[str]) -> dict[str, list[dict]]:
            if not batch:
                return {}

            try:
                return self.provider.get_news_batch(
                    batch,
                    limit_per_symbol=per_symbol_limit,
                    start=start,
                ) or {}
            except Exception as e:
                # If one symbol poisons the whole batch, split until isolated.
                if len(batch) == 1:
                    return {batch[0]: [{"__fetch_error__": f"{type(e).__name__}: {e}"}]}

                mid = len(batch) // 2
                left = _fetch_news_resilient(batch[:mid])
                right = _fetch_news_resilient(batch[mid:])
                merged = {}
                merged.update(left)
                merged.update(right)
                return merged

        raw_map: dict[str, list[dict]] = _fetch_news_resilient(eligible_syms)

        out: dict[str, CatalystBundle] = {}

        for s in ineligible_syms:
            out[s] = _empty_bundle(s, error='symbol_ineligible_for_news')

        for s in eligible_syms:
            rows = raw_map.get(s, [])

            if rows and isinstance(rows, list) and isinstance(rows[0], dict) and '__fetch_error__' in rows[0]:
                out[s] = _empty_bundle(s, error=str(rows[0]['__fetch_error__']))
            else:
                out[s] = self._score_symbol(s, rows, now=now)

        return out

    def _score_symbol(self, symbol: str, rows: list[dict], *, now: datetime) -> CatalystBundle:
        articles: list[CatalystArticle] = []
        source_set: set[str] = set()
        freshness_hours: float | None = None
        weighted = 0.0
        weight_sum = 0.0
        abs_strength = 0.0
        tag_set: set[str] = set()

        for r in rows or []:
            headline = str(r.get('headline') or r.get('title') or '').strip()
            summary = (r.get('summary') or r.get('content') or None)
            source = None
            src = r.get('source')
            if isinstance(src, dict):
                source = str(src.get('name') or src.get('id') or '').strip() or None
            elif src is not None:
                source = str(src).strip() or None
            created = _parse_dt(r.get('created_at') or r.get('updated_at') or r.get('timestamp'))
            h = _hours_since(created, now)
            if h is not None:
                freshness_hours = h if freshness_hours is None else min(freshness_hours, h)
            rec_w = _recency_weight(h)
            text = f"{headline} {summary or ''}".strip()
            kscore = _keyword_score(text)
            # small source/structure bonus for actual article presence
            base = 0.08 if headline else 0.0
            art_score = max(-1.0, min(1.0, (kscore / 2.5)))
            composite = art_score + (base if art_score >= 0 else -base * 0.4)
            composite = max(-1.0, min(1.0, composite))
            tags = _tags_for_text(text)
            for t in tags:
                tag_set.add(t)
            if source:
                source_set.add(source.lower())
            weighted += composite * rec_w
            weight_sum += rec_w
            abs_strength += abs(composite) * rec_w
            articles.append(CatalystArticle(
                symbol=symbol,
                headline=headline,
                summary=(str(summary)[:500] if summary else None),
                source=source,
                created_at=(created.isoformat() if created else None),
                url=(str(r.get('url')).strip() if r.get('url') else None),
                score=float(composite),
                recency_weight=float(rec_w),
                tags=tags,
            ))

        if not articles:
            return CatalystBundle(
                symbol=symbol,
                score=0.0,
                confidence=0.0,
                strength=0.0,
                article_count=0,
                source_count=0,
                freshness_hours=None,
                tags=[],
                top_headlines=[],
                articles=[],
            )

        directional = (weighted / weight_sum) if weight_sum > 0 else 0.0
        directional = max(-1.0, min(1.0, directional))
        strength = min(1.0, abs_strength / max(1.0, weight_sum))
        article_factor = min(1.0, len(articles) / 4.0)
        source_factor = min(1.0, len(source_set) / 3.0)
        recency_factor = _recency_weight(freshness_hours)
        confidence = 100.0 * (0.45 * recency_factor + 0.30 * strength + 0.15 * article_factor + 0.10 * source_factor)
        # boost confidence for clear catalyst tags
        if tag_set:
            confidence = min(100.0, confidence + min(12.0, 2.5 * len(tag_set)))

        # Rank articles by |score|*recency then clip
        articles.sort(key=lambda a: (abs(a.score) * a.recency_weight, a.recency_weight), reverse=True)
        top_headlines = [a.headline for a in articles[:3] if a.headline]

        return CatalystBundle(
            symbol=symbol,
            score=float(directional),
            confidence=float(round(confidence, 2)),
            strength=float(round(strength, 4)),
            article_count=len(articles),
            source_count=len(source_set),
            freshness_hours=(None if freshness_hours is None else float(round(freshness_hours, 2))),
            tags=sorted(tag_set),
            top_headlines=top_headlines,
            articles=articles[:5],
        )
