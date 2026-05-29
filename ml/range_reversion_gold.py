from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import joblib

RTH_MINUTES = 390.0
EPS = 1e-12
ID_COLS = {"symbol", "day", "label", "ts_utc"}

@dataclass(frozen=True)
class RRGoldBundle:
    model: object
    calibrator: object
    feature_columns: List[str]
    clip_bounds: Dict[str, Tuple[float, float]]

class RangeReversionGoldScorer:
    """
    Loads models/range_reversion_gold.pkl and produces calibrated probabilities.

    Replicates training-time feature engineering + train-fit winsorization + isotonic calibration.
    """

    def __init__(self, model_path: Union[str, Path] = Path("models/range_reversion_gold.pkl")) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RR gold model not found: {self.model_path.resolve()}")

        raw = joblib.load(self.model_path)
        for k in ("model", "calibrator", "feature_columns", "clip_bounds"):
            if k not in raw:
                raise RuntimeError(f"Invalid RR gold bundle (missing {k}). Keys={list(raw.keys())}")

        self.bundle = RRGoldBundle(
            model=raw["model"],
            calibrator=raw["calibrator"],
            feature_columns=list(raw["feature_columns"]),
            clip_bounds=dict(raw["clip_bounds"]),
        )

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for c in out.columns:
            if c in ID_COLS:
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce")

        if "zscore" in out.columns:
            out["abs_zscore"] = out["zscore"].abs()
            out["zscore_sq"] = out["zscore"] * out["zscore"]

        if "tod_min" in out.columns:
            ang = (2.0 * math.pi) * (out["tod_min"].clip(0, RTH_MINUTES) / RTH_MINUTES)
            out["tod_sin"] = np.sin(ang)
            out["tod_cos"] = np.cos(ang)

        if "volume" in out.columns:
            out["log_volume"] = np.log1p(out["volume"].clip(lower=0))

        if "transactions" in out.columns:
            out["log_transactions"] = np.log1p(out["transactions"].clip(lower=0))

        if "relvol5" in out.columns and "relvol15" in out.columns:
            out["relvol_ratio"] = out["relvol5"] / (out["relvol15"] + EPS)

        if "slope" in out.columns:
            out["abs_slope"] = out["slope"].abs()

        if "dist_to_entry_sig" in out.columns:
            out["entry_proximity"] = -out["dist_to_entry_sig"].abs()

        return out

    def _apply_clip_bounds(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, (lo, hi) in self.bundle.clip_bounds.items():
            if c in out.columns:
                out[c] = out[c].clip(lower=lo, upper=hi)
        return out

    def predict_proba(self, df_base: pd.DataFrame) -> np.ndarray:
        df_feat = self._engineer_features(df_base)

        missing = [c for c in self.bundle.feature_columns if c not in df_feat.columns]
        if missing:
            raise RuntimeError(f"Missing required RR feature columns: {missing}")

        X = df_feat[self.bundle.feature_columns].astype(float)
        X = self._apply_clip_bounds(X)

        p_raw = self.bundle.model.predict_proba(X)[:, 1]
        cal = self.bundle.calibrator
        if not hasattr(cal, "transform"):
            raise RuntimeError(f"Unsupported calibrator type: {type(cal)}")
        p = cal.transform(p_raw)
        return np.asarray(p, dtype=float)
