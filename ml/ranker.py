from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import numpy as np
import pandas as pd
import pytz
import joblib
import logging
logger = logging.getLogger(__name__)

from providers.base import BarsRequest
from providers.alpaca_provider import AlpacaProvider


def _env_bool(name: str, default: bool = False) -> bool:
    import os
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1","true","yes","y","on"}


_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = (
    _MODELS_DIR / "model_a_liquid.pkl"
    if (_MODELS_DIR / "model_a_liquid.pkl").exists()
    else (_MODELS_DIR / "orb_ranker.pkl")
)


@dataclass
class RankerConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    strict_ml: bool = _env_bool("ORB_STRICT_ML", True)


class ORBRanker:
    """Optional ML ranker that scores already-valid candidates.

    Design goals:
    - Uses only real Massive market data via REST (no sample dictionaries).
    - If data is missing, raise real exceptions during training. In scoring, we fail per-symbol
      and simply omit score for that symbol.
    - Model is a real trained sklearn model persisted to a .pkl via joblib.
    """

    def __init__(self, cfg: Optional[RankerConfig] = None, provider: Optional[AlpacaProvider] = None):
        self.cfg = cfg or RankerConfig()
        self.provider = provider or AlpacaProvider()
        self.model = None
        self.feature_names: List[str] = []
        # Per-symbol scoring failures (real exceptions). Populated during score_candidates().
        self.failures: List[dict] = []

    def load(self) -> bool:
        path = Path(self.cfg.model_path)
        if not path.exists():
            if getattr(self.cfg, "strict_ml", False):
                raise FileNotFoundError(f"ORB ML model not found: {path.resolve()}")
            try:
                logger.warning("ML disabled: model not found at %s", path)
            except Exception:
                pass
            return False
        blob = joblib.load(path)
        if not isinstance(blob, dict):
            raise TypeError(f"Expected dict from joblib.load({path}), got {type(blob)}")
        if "model" not in blob or "feature_names" not in blob:
            raise KeyError(f"Model file missing keys; found: {list(blob.keys())}")
        self.model = blob["model"]
        self.feature_names = list(blob["feature_names"])
        if self.model is None:
            raise ValueError("Loaded model is None")
        if not self.feature_names:
            raise ValueError("Loaded feature_names is empty")
        return True

    def score_candidates(self, symbols: List[str]) -> Dict[str, float]:
        """Return ML score per symbol. Symbols without features/data are omitted."""
        if self.model is None:
            if not self.load():
                return {}

        rows = []
        kept = []
        last_err = None
        for sym in symbols:
            try:
                feats = self._features_for_symbol(sym)
                rows.append(feats)
                kept.append(sym)
            except Exception as e:
                last_err = e
                try:
                    self.failures.append({
                        "symbol": sym,
                        "stage": "ml_features",
                        "error": f"{type(e).__name__}: {e}",
                    })
                except Exception:
                    pass
                continue
        if not rows:
            try:
                logger.warning("ML produced 0 feature rows; last_error=%r", last_err)
            except Exception:
                pass
            return {}

        X = pd.DataFrame(rows)
        # align — missing features filled with 0.0 (neutral); log for visibility
        missing = [f for f in self.feature_names if f not in X.columns]
        if missing:
            try:
                logger.warning("ML feature gap (filled with 0.0): %s", missing)
            except Exception:
                pass
            for f in missing:
                X[f] = np.nan
        X = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        proba = self.model.predict_proba(X)[:, 1]
        return {sym: float(p) for sym, p in zip(kept, proba)}

    # ------------------------
    # Feature engineering
    # ------------------------
    def _features_for_symbol(self, symbol: str) -> Dict[str, float]:
        """Compute lightweight features available on any day.

        We do NOT require OR window here; the scanner already filtered to candidates with OR window.
        Features use:
          - Daily (6mo) for trend/volatility/liquidity
          - Intraday (1d, 1m) for early liquidity and volatility
        """
        daily = self.provider.get_daily_history(symbol, period="6mo").sort_index()
        if daily is None or daily.empty:
            raise RuntimeError(f"No daily history for {symbol}")

        close = daily["Close"].astype(float)
        vol = daily["Volume"].astype(float)

        # Trend (continuous): sma20/sma50 - 1 (matches A/B liquid feature definition)
        _sma20_raw = close.rolling(20).mean().iloc[-1]
        _sma50_raw = close.rolling(50).mean().iloc[-1]
        if pd.isna(_sma20_raw) or pd.isna(_sma50_raw):
            raise RuntimeError(f"Insufficient daily history for SMA features on {symbol}")
        sma20 = float(_sma20_raw)
        sma50 = float(_sma50_raw)
        trend_20_50 = float(sma20 / sma50 - 1.0)

        # Volatility proxy
        ret = close.pct_change()
        _vol20_raw = ret.rolling(20).std().iloc[-1]
        if pd.isna(_vol20_raw):
            raise RuntimeError(f"Insufficient daily history for vol20 on {symbol}")
        vol20 = float(_vol20_raw)

        # Liquidity (avg $ volume 20d)
        _avg20_vol_raw = vol.rolling(20).mean().iloc[-1]
        if pd.isna(_avg20_vol_raw):
            raise RuntimeError(f"Insufficient daily history for avg20_vol on {symbol}")
        avg20_vol = float(_avg20_vol_raw)
        last_close = float(close.iloc[-1])
        avg20_dollar_vol = float(avg20_vol * last_close)

        # ATR14% (daily) — required by the liquid model
        high = daily["High"].astype(float)
        low = daily["Low"].astype(float)
        prev_close_series = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close_series).abs(), (low - prev_close_series).abs()],
            axis=1,
        ).max(axis=1)
        _atr14_raw = tr.rolling(14).mean().iloc[-1]
        if pd.isna(_atr14_raw):
            raise RuntimeError(f"Insufficient daily history for ATR14 on {symbol}")
        atr14 = float(_atr14_raw)
        atr14_pct = float((atr14 / last_close) if last_close > 0 else 0.0)

        # Recent momentum
        mom5 = float((close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0)
        mom20 = float((close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0)
        # Intraday 1m "most recent complete session". Robust across weekends/holidays.
        # ML scoring should be fast-fail: if a symbol can't fetch in a few seconds, skip it
        # rather than stalling an entire scan.
        intraday_raw = self.provider.get_bars(
            BarsRequest(symbol=symbol, interval="1m", period="5d", include_prepost=False),
            timeout_s=min(
                10,
                int(
                    getattr(self.provider, "request_timeout_seconds", 20)
                    or os.getenv("ORB_ML_INTRADAY_TIMEOUT_S", "20")
                ),
            ),
        )
        if intraday_raw is None or intraday_raw.empty:
            raise RuntimeError(f"No intraday bars for {symbol}")
        intraday_raw = intraday_raw.sort_index()

        # Use the most recent trading date present in intraday data, in New York time.
        ny = pytz.timezone("America/New_York")
        if not isinstance(intraday_raw.index, pd.DatetimeIndex):
            raise RuntimeError(f"Intraday index is not DatetimeIndex for {symbol}")
        if intraday_raw.index.tz is None:
            # Provider should return tz-aware UTC; assume UTC if not.
            intraday_raw.index = intraday_raw.index.tz_localize(pytz.UTC)
        intraday_raw = intraday_raw.tz_convert(ny)
        dates = intraday_raw.index.normalize()
        last_date = dates[-1]
        intraday = intraday_raw[dates == last_date]
        if intraday is None or intraday.empty:
            raise RuntimeError(f"No intraday session slice for {symbol}")

        intr_open = intraday["Open"].astype(float) if "Open" in intraday.columns else intraday["Close"].astype(float)
        intr_high = intraday["High"].astype(float) if "High" in intraday.columns else intraday["Close"].astype(float)
        intr_low = intraday["Low"].astype(float) if "Low" in intraday.columns else intraday["Close"].astype(float)
        intr_close = intraday["Close"].astype(float)
        intr_vol = intraday["Volume"].astype(float)

        day_vol = float(intr_vol.sum())
        day_dollar_vol = day_vol * float(intr_close.iloc[-1])

        first = float(intr_close.iloc[0])
        last = float(intr_close.iloc[-1])
        day_ret = float((last / first - 1.0) if first > 0 else 0.0)

        # Gap pct (today open vs prev close) as a fraction (not percent)
        prev_close = float(close.iloc[-1] or 0.0)
        today_open = float(intr_open.iloc[0] or 0.0)
        gap_pct = float((today_open / prev_close - 1.0) if prev_close > 0 else 0.0)

        # OR range pct from the first 5 minutes (5-minute ORB), as a fraction.
        early_5 = intraday.iloc[:5]
        if early_5.empty:
            raise RuntimeError(f"Not enough early 1m bars for {symbol}")
        or_high = float(early_5["High"].astype(float).max()) if "High" in early_5.columns else float(early_5["Close"].astype(float).max())
        or_low = float(early_5["Low"].astype(float).min()) if "Low" in early_5.columns else float(early_5["Close"].astype(float).min())
        or_open = float(intr_open.iloc[0] or 0.0)
        if or_open <= 0:
            raise RuntimeError(f"Invalid OR open for {symbol}")
        or_range_pct = float((or_high - or_low) / or_open)

        # Relative volume features vs avg20 daily volume (matches training)
        if avg20_vol <= 0:
            raise RuntimeError(f"avg20_vol unavailable for {symbol}; cannot compute relvol")
        relvol5 = float(early_5["Volume"].astype(float).sum() / avg20_vol) if len(early_5) else 0.0
        early_15 = intraday.iloc[:15]
        relvol15 = float(early_15["Volume"].astype(float).sum() / avg20_vol) if len(early_15) else 0.0


        return {
            # Features used by the liquid A/B model
            "trend_20_50": float(trend_20_50),
            "vol20": float(vol20),
            "avg20_dollar_vol": float(avg20_dollar_vol),
            "mom5": float(mom5),
            "mom20": float(mom20),
            "gap_pct": float(gap_pct),
            "atr14_pct": float(atr14_pct),
            "or_range_pct": float(or_range_pct),
            "relvol5": float(relvol5),
            "relvol15": float(relvol15),

            # Extra diagnostics (older models may still use these)
            "day_dollar_vol": float(day_dollar_vol),
            "day_ret": float(day_ret),
        }
