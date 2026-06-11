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


def _fetch_market_context(provider: AlpacaProvider, ny: Any) -> Dict[str, float]:
    """Fetch SPY + QQQ intraday bars once and return context features.

    Returns dict with:
      - spy_intraday_ret: SPY % move from open to last bar
      - qqq_intraday_ret: QQQ % move from open to last bar
      - spy_qqq_divergence: abs(spy_ret - qqq_ret) — measures sector rotation strength
      - market_up: 1.0 if both SPY and QQQ positive, 0.0 otherwise

    Raises if SPY or QQQ data is unavailable — market context is not optional.
    """
    def _intraday_ret(sym: str) -> float:
        bars = provider.get_bars(BarsRequest(symbol=sym, interval="1m", period="1d", include_prepost=False))
        if bars is None or bars.empty:
            raise RuntimeError(f"{sym} intraday bars empty")
        bars = bars.sort_index()
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize(pytz.UTC)
        bars = bars.tz_convert(ny)
        dates = bars.index.normalize()
        last_date = dates[-1]
        session = bars[dates == last_date]
        if session is None or session.empty:
            raise RuntimeError(f"{sym} no bars for last date")
        open_col = "Open" if "Open" in session.columns else "open"
        close_col = "Close" if "Close" in session.columns else "close"
        first_open = float(session[open_col].iloc[0])
        last_close = float(session[close_col].iloc[-1])
        if first_open <= 0:
            raise RuntimeError(f"{sym} invalid open price {first_open}")
        return float((last_close / first_open - 1.0) * 100.0)

    spy_ret = _intraday_ret("SPY")
    qqq_ret = _intraday_ret("QQQ")

    return {
        "spy_intraday_ret": spy_ret,
        "qqq_intraday_ret": qqq_ret,
        "spy_qqq_divergence": float(abs(spy_ret - qqq_ret)),
        "market_up": 1.0 if spy_ret > 0 and qqq_ret > 0 else 0.0,
    }


class ORBRanker:
    """ML ranker that scores already-valid ORB candidates.

    Design goals:
    - Uses only real market data via REST (no sample dictionaries).
    - If data is missing, raise real exceptions. In scoring, fail per-symbol
      and omit score for that symbol.
    - Model is a real trained sklearn/lgbm model persisted to .pkl via joblib.
    - Market context (SPY/QQQ) is fetched ONCE per batch — not per symbol.
    """

    def __init__(self, cfg: Optional[RankerConfig] = None, provider: Optional[AlpacaProvider] = None):
        self.cfg = cfg or RankerConfig()
        self.provider = provider or AlpacaProvider()
        self.model = None
        self.feature_names: List[str] = []
        self.failures: List[dict] = []
        self._market_ctx: Dict[str, float] | None = None

    def load(self) -> bool:
        path = Path(self.cfg.model_path)
        if not path.exists():
            if getattr(self.cfg, "strict_ml", False):
                raise FileNotFoundError(f"ORB ML model not found: {path.resolve()}")
            logger.warning("ML disabled: model not found at %s", path)
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

    def _load_market_context(self) -> None:
        """Fetch SPY/QQQ context once per ranker instance."""
        ny = pytz.timezone("America/New_York")
        try:
            self._market_ctx = _fetch_market_context(self.provider, ny)
        except Exception as e:
            raise RuntimeError(f"Market context fetch failed (SPY/QQQ required): {e}") from e

    def score_candidates(self, symbols: List[str]) -> Dict[str, float]:
        """Return ML score per symbol. Symbols without features/data are omitted."""
        if self.model is None:
            if not self.load():
                return {}

        # Fetch market context once for the entire batch
        if self._market_ctx is None:
            self._load_market_context()

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
                self.failures.append({
                    "symbol": sym,
                    "stage": "ml_features",
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
        if not rows:
            logger.warning("ML produced 0 feature rows; last_error=%r", last_err)
            return {}

        X = pd.DataFrame(rows)
        missing = [f for f in self.feature_names if f not in X.columns]
        if missing:
            logger.warning("ML feature gap (missing entirely — not filled): %s", missing)
            for f in missing:
                X[f] = np.nan
        X = X[self.feature_names].replace([np.inf, -np.inf], np.nan)
        # NaN in features = real data failure; propagate as NaN score rather than silently filling 0
        # LightGBM and sklearn handle NaN natively via fillna(0) only when unavoidable
        X = X.fillna(0.0)

        proba = self.model.predict_proba(X)[:, 1]
        return {sym: float(p) for sym, p in zip(kept, proba)}

    # ------------------------
    # Feature engineering
    # ------------------------
    def _features_for_symbol(self, symbol: str) -> Dict[str, float]:
        """Compute all features for a single symbol.

        Features:
          Symbol-level (daily):
            trend_20_50, vol20, avg20_dollar_vol, mom5, mom20, atr14_pct, gap_pct
          Symbol-level (intraday):
            or_range_pct, relvol5, relvol15
          Anchored VWAP:
            prev_close_vwap_delta_pct: (vwap - prev_close) / prev_close * 100
            or_midpoint_vs_vwap_pct:   (or_mid - vwap) / vwap * 100
          Market context (batch-shared from SPY/QQQ):
            spy_intraday_ret, qqq_intraday_ret, spy_qqq_divergence, market_up
        """
        daily = self.provider.get_daily_history(symbol, period="6mo").sort_index()
        if daily is None or daily.empty:
            raise RuntimeError(f"No daily history for {symbol}")

        close = daily["Close"].astype(float)
        vol = daily["Volume"].astype(float)
        high = daily["High"].astype(float)
        low = daily["Low"].astype(float)

        _sma20 = close.rolling(20).mean().iloc[-1]
        _sma50 = close.rolling(50).mean().iloc[-1]
        if pd.isna(_sma20) or pd.isna(_sma50):
            raise RuntimeError(f"Insufficient daily history for SMA features on {symbol}")
        trend_20_50 = float(_sma20 / _sma50 - 1.0)

        ret = close.pct_change()
        _vol20 = ret.rolling(20).std().iloc[-1]
        if pd.isna(_vol20):
            raise RuntimeError(f"Insufficient daily history for vol20 on {symbol}")
        vol20 = float(_vol20)

        _avg20_vol = vol.rolling(20).mean().iloc[-1]
        if pd.isna(_avg20_vol):
            raise RuntimeError(f"Insufficient daily history for avg20_vol on {symbol}")
        avg20_vol = float(_avg20_vol)
        last_close = float(close.iloc[-1])
        avg20_dollar_vol = float(avg20_vol * last_close)

        prev_close_series = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close_series).abs(), (low - prev_close_series).abs()],
            axis=1,
        ).max(axis=1)
        _atr14 = tr.rolling(14).mean().iloc[-1]
        if pd.isna(_atr14):
            raise RuntimeError(f"Insufficient daily history for ATR14 on {symbol}")
        atr14_pct = float(float(_atr14) / last_close) if last_close > 0 else None
        if atr14_pct is None:
            raise RuntimeError(f"Invalid last_close for {symbol}")

        mom5 = float((close.iloc[-1] / close.iloc[-6] - 1.0)) if len(close) >= 6 else None
        mom20 = float((close.iloc[-1] / close.iloc[-21] - 1.0)) if len(close) >= 21 else None
        if mom5 is None:
            raise RuntimeError(f"Insufficient history for mom5 on {symbol}")
        if mom20 is None:
            raise RuntimeError(f"Insufficient history for mom20 on {symbol}")

        # Intraday bars
        ny = pytz.timezone("America/New_York")
        intraday_raw = self.provider.get_bars(
            BarsRequest(symbol=symbol, interval="1m", period="5d", include_prepost=False),
            timeout_s=min(10, int(getattr(self.provider, "request_timeout_seconds", 20)
                          or os.getenv("ORB_ML_INTRADAY_TIMEOUT_S", "20"))),
        )
        if intraday_raw is None or intraday_raw.empty:
            raise RuntimeError(f"No intraday bars for {symbol}")
        intraday_raw = intraday_raw.sort_index()
        if not isinstance(intraday_raw.index, pd.DatetimeIndex):
            raise RuntimeError(f"Intraday index is not DatetimeIndex for {symbol}")
        if intraday_raw.index.tz is None:
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

        prev_close_val = float(close.iloc[-1])
        today_open = float(intr_open.iloc[0])
        if prev_close_val <= 0:
            raise RuntimeError(f"Invalid prev_close for {symbol}")
        if today_open <= 0:
            raise RuntimeError(f"Invalid today_open for {symbol}")
        gap_pct = float((today_open / prev_close_val - 1.0))

        early_5 = intraday.iloc[:5]
        if early_5.empty:
            raise RuntimeError(f"Not enough early 1m bars for {symbol}")
        or_high = float(early_5["High"].astype(float).max()) if "High" in early_5.columns else float(early_5["Close"].astype(float).max())
        or_low = float(early_5["Low"].astype(float).min()) if "Low" in early_5.columns else float(early_5["Close"].astype(float).min())
        or_open = float(intr_open.iloc[0])
        if or_open <= 0:
            raise RuntimeError(f"Invalid OR open for {symbol}")
        or_range_pct = float((or_high - or_low) / or_open)
        or_mid = (or_high + or_low) / 2.0

        if avg20_vol <= 0:
            raise RuntimeError(f"avg20_vol unavailable for {symbol}")
        relvol5 = float(early_5["Volume"].astype(float).sum() / avg20_vol)
        early_15 = intraday.iloc[:15]
        relvol15 = float(early_15["Volume"].astype(float).sum() / avg20_vol)

        # VWAP (intraday, session reset)
        if all(c in intraday.columns for c in ("High", "Low", "Close", "Volume")):
            tp = (intr_high + intr_low + intr_close) / 3.0
            vwap_val = float((tp * intr_vol).cumsum().iloc[-1] / intr_vol.cumsum().iloc[-1])
            if vwap_val <= 0:
                raise RuntimeError(f"Invalid VWAP for {symbol}")
        else:
            raise RuntimeError(f"Missing OHLCV columns for VWAP on {symbol}")

        # Anchored VWAP features
        prev_close_vwap_delta_pct = float((vwap_val - prev_close_val) / prev_close_val * 100.0)
        or_midpoint_vs_vwap_pct = float((or_mid - vwap_val) / vwap_val * 100.0)

        # Market context features (cached at batch level)
        ctx = self._market_ctx or {}

        return {
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
            # Anchored VWAP
            "prev_close_vwap_delta_pct": float(prev_close_vwap_delta_pct),
            "or_midpoint_vs_vwap_pct": float(or_midpoint_vs_vwap_pct),
            # Market context (same for all symbols in batch)
            "spy_intraday_ret": float(ctx.get("spy_intraday_ret", 0.0)),
            "qqq_intraday_ret": float(ctx.get("qqq_intraday_ret", 0.0)),
            "spy_qqq_divergence": float(ctx.get("spy_qqq_divergence", 0.0)),
            "market_up": float(ctx.get("market_up", 0.0)),
            # Diagnostics
            "day_dollar_vol": float(float(intr_vol.sum()) * float(intr_close.iloc[-1])),
            "day_ret": float((float(intr_close.iloc[-1]) / float(intr_close.iloc[0]) - 1.0) if float(intr_close.iloc[0]) > 0 else 0.0),
        }
