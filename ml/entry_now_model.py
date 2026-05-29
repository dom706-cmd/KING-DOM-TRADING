from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np


@dataclass(frozen=True)
class EntryNowModelInfo:
    path: str
    kind: str
    created_at: str
    pos_rate: float
    auc: float


_MODEL_CACHE: dict[str, tuple[float, int, Any]] = {}


def load_entry_now_model(path: str = "models/entry_now_30m.pkl") -> tuple[Any, list[str], float]:
    """Load entry-now model bundle. Cached by mtime/size."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Entry-now model not found: {path}")
    st = p.stat()
    key = str(p.resolve())
    cached = _MODEL_CACHE.get(key)
    if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
        bundle = cached[2]
    else:
        bundle = joblib.load(p)
        _MODEL_CACHE[key] = (st.st_mtime, st.st_size, bundle)

    if not isinstance(bundle, dict) or "model" not in bundle or "feature_names" not in bundle:
        raise RuntimeError("Invalid entry-now model bundle format")
    model = bundle["model"]
    feats = list(bundle["feature_names"])
    fillna = float(bundle.get("fillna_value", -999.0))
    return model, feats, fillna


def score_entry_now(features: Dict[str, float], *, model_path: str = "models/entry_now_30m.pkl") -> float:
    model, cols, fillna = load_entry_now_model(model_path)
    x = []
    for c in cols:
        v = features.get(c, np.nan)
        try:
            v = float(v)
        except Exception:
            v = np.nan
        if not np.isfinite(v):
            v = fillna
        x.append(v)
    X = np.array([x], dtype=float)
    proba = model.predict_proba(X)[0, 1]
    return float(proba)
