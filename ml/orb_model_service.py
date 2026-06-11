from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ml.model_registry import env_orb_strict_ml, resolve_orb_bucket_path, resolve_orb_outcomes_path
from ml.ranker import ORBRanker, RankerConfig


_STRATEGY_MODEL_NAMES: dict[str, str] = {
    "orb": "orb_ranker_outcomes.pkl",
    "parabolic": "parabolic_ranker_outcomes.pkl",
    "atr_expansion": "atr_expansion_ranker_outcomes.pkl",
    "eod_momentum": "eod_momentum_ranker_outcomes.pkl",
}


def _outcomes_model_is_usable(strategy: str = "orb") -> Path | None:
    """Return path to the strategy-specific outcomes model if usable (AUC >= 0.52, n >= 30).

    Falls back to the generic orb outcomes model if no strategy-specific one exists.
    """
    import joblib

    def _check(path: Path) -> Path | None:
        if not path.exists():
            return None
        try:
            blob = joblib.load(path)
            if not isinstance(blob, dict) or "model" not in blob or "feature_names" not in blob:
                return None
            cv_score = blob.get("cv_roc_auc_mean")
            if cv_score is not None and float(cv_score) < 0.52:
                return None
            if int(blob.get("n_samples", 0)) < 30:
                return None
            return path
        except Exception:
            return None

    _models_dir = resolve_orb_outcomes_path().parent

    # Try strategy-specific model first
    strat_key = (strategy or "orb").lower().strip()
    filename = _STRATEGY_MODEL_NAMES.get(strat_key)
    if filename:
        p = _check(_models_dir / filename)
        if p is not None:
            return p

    # Fall back to generic orb outcomes model
    return _check(resolve_orb_outcomes_path())


class OrbModelSelectionError(RuntimeError):
    """Base error for strict ORB model routing/scoring failures."""


class OrbModelMissingError(FileNotFoundError, OrbModelSelectionError):
    """Raised when a required strict ORB model file is missing."""


class OrbModelLoadError(OrbModelSelectionError):
    """Raised when a strict ORB model cannot be loaded/validated."""


class OrbModelScoreError(OrbModelSelectionError):
    """Raised when a strict ORB model cannot score its assigned symbols."""


def bucket_name_for_price(price: float, *, rvol: float | None = None) -> str:
    from ml.model_registry import resolve_orb_parabolic_path
    if rvol is not None and float(rvol) >= 10.0 and resolve_orb_parabolic_path().exists():
        return "parabolic"
    return "under30" if float(price) < 30.0 else "liquid"


def bucket_model_path_for_price(price: float, *, rvol: float | None = None) -> Path:
    return Path(resolve_orb_bucket_path(float(price), rvol=rvol)).resolve()


def _normalize_candidate_ref(item: Any) -> dict[str, Any]:
    symbol = str(getattr(item, "symbol", "") or "").strip().upper()
    if not symbol:
        raise ValueError("candidate missing symbol")

    raw_price = getattr(item, "last_price", None)
    if raw_price is None:
        raise ValueError(f"candidate {symbol} missing last_price")

    price = float(raw_price)
    if price <= 0:
        raise ValueError(f"candidate {symbol} has invalid last_price={price}")

    raw_rvol = getattr(item, "rvol", None) or getattr(item, "rvol_hint", None)
    rvol = float(raw_rvol) if raw_rvol is not None else None

    bucket = bucket_name_for_price(price, rvol=rvol)
    model_path = bucket_model_path_for_price(price, rvol=rvol)
    return {
        "symbol": symbol,
        "price": price,
        "rvol": rvol,
        "bucket": bucket,
        "model_path": model_path,
    }


def score_orb_candidates(candidates: Iterable[Any], *, provider: Any, strategy: str = "orb") -> dict[str, Any]:
    normalized = [_normalize_candidate_ref(c) for c in candidates]
    strict = bool(env_orb_strict_ml())

    # Use strategy-specific outcomes model if available and validated
    outcomes_path = _outcomes_model_is_usable(strategy=strategy)

    grouped: dict[tuple[str, Path], list[str]] = {}
    bucket_by_symbol: dict[str, str] = {}
    model_path_by_symbol: dict[str, str] = {}

    for item in normalized:
        symbol = str(item["symbol"])
        if outcomes_path is not None:
            bucket = "outcomes"
            model_path = outcomes_path.resolve()
        else:
            bucket = str(item["bucket"])
            model_path = Path(item["model_path"]).resolve()
        bucket_by_symbol[symbol] = bucket
        model_path_by_symbol[symbol] = str(model_path)
        grouped.setdefault((bucket, model_path), []).append(symbol)

    scores: dict[str, float] = {}
    failures: list[dict[str, Any]] = []

    for (bucket, model_path), symbols in grouped.items():
        if strict and not model_path.exists():
            raise OrbModelMissingError(
                f"Strict ORB ML requires model for bucket='{bucket}' at {model_path}"
            )

        try:
            ranker = ORBRanker(
                cfg=RankerConfig(model_path=model_path, strict_ml=strict),
                provider=provider,
            )
            ranker.load()
        except FileNotFoundError as e:
            raise OrbModelMissingError(
                f"Strict ORB ML requires model for bucket='{bucket}' at {model_path}"
            ) from e
        except Exception as e:
            raise OrbModelLoadError(
                f"Failed loading ORB ML bucket='{bucket}' from {model_path}: {type(e).__name__}: {e}"
            ) from e

        try:
            batch_scores = ranker.score_candidates(symbols)
        except Exception as e:
            raise OrbModelScoreError(
                f"Failed scoring ORB ML bucket='{bucket}' from {model_path}: {type(e).__name__}: {e}"
            ) from e

        failures.extend(list(getattr(ranker, "failures", []) or []))
        for symbol, score in batch_scores.items():
            scores[str(symbol)] = float(score)

        for symbol in symbols:
            if symbol not in batch_scores:
                failures.append(
                    {
                        "symbol": symbol,
                        "stage": "ml_score_missing",
                        "bucket": bucket,
                        "model_path": str(model_path),
                        "error": "strict_orb_ml_scored_no_output",
                    }
                )

    return {
        "scores": scores,
        "bucket_by_symbol": bucket_by_symbol,
        "model_path_by_symbol": model_path_by_symbol,
        "failures": failures,
        "strict": strict,
    }


def score_orb_symbol(symbol: str, *, last_price: float, provider: Any, rvol: float | None = None) -> dict[str, Any]:
    class _One:
        def __init__(self, symbol: str, last_price: float, rvol: float | None) -> None:
            self.symbol = symbol
            self.last_price = last_price
            self.rvol = rvol

    out = score_orb_candidates([_One(symbol=symbol, last_price=last_price, rvol=rvol)], provider=provider)
    sym = str(symbol).strip().upper()
    return {
        "symbol": sym,
        "score": out["scores"].get(sym),
        "bucket": out["bucket_by_symbol"].get(sym),
        "model_path": out["model_path_by_symbol"].get(sym),
        "failures": [f for f in out["failures"] if str(f.get("symbol", "")).upper() == sym],
        "strict": out["strict"],
    }
