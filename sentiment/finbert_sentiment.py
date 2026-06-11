"""FinBERT-based sentiment scorer for financial headlines.

Uses ProsusAI/finbert — fine-tuned on financial news, analyst reports, and earnings
commentary. Outputs positive/negative/neutral probabilities.

Score = positive_prob - negative_prob  →  range [-1.0, 1.0]

Model is loaded once at first use and cached for the process lifetime.
No fallbacks: if the model cannot load or score, raise.
"""
from __future__ import annotations

import math
import threading
from typing import List

_MODEL_ID = "ProsusAI/finbert"
_MAX_TOKENS = 512

_lock = threading.Lock()
_pipe = None


def _load_pipeline():
    global _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        from transformers import pipeline
        _pipe = pipeline(
            "text-classification",
            model=_MODEL_ID,
            top_k=None,
            truncation=True,
            max_length=_MAX_TOKENS,
            device=-1,  # CPU; set to 0 for GPU if available
        )
        return _pipe


def score_headline(text: str) -> float:
    """Score a single headline. Returns float in [-1.0, 1.0]."""
    if not text or not text.strip():
        raise ValueError("headline text required")
    pipe = _load_pipeline()
    results = pipe(text[:_MAX_TOKENS])[0]
    probs = {r["label"].lower(): float(r["score"]) for r in results}
    pos = probs.get("positive", 0.0)
    neg = probs.get("negative", 0.0)
    score = pos - neg
    if not math.isfinite(score):
        raise ValueError(f"FinBERT returned non-finite score for headline: {text[:80]!r}")
    return float(score)


def score_headlines(headlines: List[str]) -> float:
    """Score a list of headlines, return average score in [-1.0, 1.0].
    Raises if list is empty or all headlines are empty.
    """
    clean = [h.strip() for h in headlines if h and h.strip()]
    if not clean:
        raise ValueError("No non-empty headlines to score")
    scores = [score_headline(h) for h in clean[:20]]
    return float(sum(scores) / len(scores))
