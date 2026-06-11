from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _repo_root() -> Path:
    # ml/ -> repo root
    return Path(__file__).resolve().parents[1]


def _models_dir() -> Path:
    return _repo_root() / "models"


def resolve_model_path(env_var: str, default_rel: str) -> Path:
    raw = (os.getenv(env_var) or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = _repo_root() / p
        return p
    return _repo_root() / default_rel


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime_utc_iso(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    st = path.stat()
    return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelStatus:
    path: str
    exists: bool
    mtime_utc: Optional[str]
    sha256: Optional[str]
    loadable_ok: Optional[bool] = None
    load_error: Optional[str] = None


def file_status(path: Path) -> ModelStatus:
    p = path.resolve()
    exists = p.exists()
    return ModelStatus(
        path=str(p),
        exists=exists,
        mtime_utc=mtime_utc_iso(p) if exists else None,
        sha256=sha256_file(p) if exists else None,
        loadable_ok=None,
        load_error=None,
    )


def _orb_validate_blob(blob: Any) -> None:
    if not isinstance(blob, dict):
        raise TypeError(f"Expected dict bundle, got {type(blob)}")
    if "model" not in blob or "feature_names" not in blob:
        raise KeyError(f"ORB bundle missing keys; keys={list(blob.keys())}")
    if blob["model"] is None:
        raise ValueError("ORB bundle 'model' is None")
    feats = blob.get("feature_names")
    if not isinstance(feats, (list, tuple)) or len(feats) == 0:
        raise ValueError("ORB bundle 'feature_names' is empty or invalid")


def _rr_validate_blob(blob: Any) -> None:
    if not isinstance(blob, dict):
        raise TypeError(f"Expected dict bundle, got {type(blob)}")
    for k in ("model", "calibrator", "feature_columns", "clip_bounds"):
        if k not in blob:
            raise KeyError(f"RR bundle missing key '{k}'; keys={list(blob.keys())}")
    if blob["model"] is None:
        raise ValueError("RR bundle 'model' is None")
    if blob["calibrator"] is None:
        raise ValueError("RR bundle 'calibrator' is None")


def load_check(path: Path, *, kind: str) -> Tuple[bool, Optional[str]]:
    """
    Best-effort validation that the model file is loadable.
    This uses joblib and inspects required keys only.
    """
    if not path.exists():
        return False, "file does not exist"
    try:
        import joblib  # local import to avoid hard dependency for status-only calls

        blob = joblib.load(path)
        if kind == "orb":
            _orb_validate_blob(blob)
        elif kind == "rr":
            _rr_validate_blob(blob)
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def get_model_status(*, include_load_check: bool = False) -> Dict[str, Any]:
    """
    Canonical status payload for UI/API.
    """
    orb_path = resolve_model_path("ORB_MODEL_PATH", "models/model_a_liquid.pkl")
    rr_path = resolve_model_path("RR_MODEL_PATH", "models/range_reversion_gold.pkl")

    strict_orb = _env_bool("ORB_STRICT_ML", True)
    strict_rr = _env_bool("RR_STRICT_ML", True)

    orb = file_status(orb_path)
    rr = file_status(rr_path)

    if include_load_check:
        ok, err = load_check(orb_path, kind="orb")
        orb = ModelStatus(**{**orb.__dict__, "loadable_ok": ok, "load_error": err})
        ok, err = load_check(rr_path, kind="rr")
        rr = ModelStatus(**{**rr.__dict__, "loadable_ok": ok, "load_error": err})

    return {
        "strict": {"orb": strict_orb, "rr": strict_rr},
        "orb": {"selected": orb.__dict__},
        "rr": {"selected": rr.__dict__},
    }


def env_orb_strict_ml() -> bool:
    return _env_bool("ORB_STRICT_ML", True)


def env_rr_strict_ml() -> bool:
    return _env_bool("RR_STRICT_ML", True)


def resolve_orb_model_a_path() -> Path:
    return resolve_model_path("ORB_MODEL_A_PATH", "models/model_a_liquid.pkl")


def resolve_orb_model_b_path() -> Path:
    return resolve_model_path("ORB_MODEL_B_PATH", "models/model_b_outlier.pkl")


def resolve_orb_parabolic_path() -> Path:
    return resolve_model_path("ORB_MODEL_PARABOLIC_PATH", "models/model_parabolic.pkl")


def resolve_orb_bucket_path(price: float, *, rvol: float | None = None) -> Path:
    para_path = resolve_orb_parabolic_path()
    if para_path.exists() and rvol is not None and float(rvol) >= 10.0:
        return para_path
    return resolve_orb_model_b_path() if float(price) < 30.0 else resolve_orb_model_a_path()


def resolve_orb_outcomes_path() -> Path:
    return resolve_model_path("ORB_MODEL_OUTCOMES_PATH", "models/orb_ranker_outcomes.pkl")


def resolve_rr_model_path() -> Path:
    return resolve_model_path("RR_MODEL_PATH", "models/range_reversion_gold.pkl")


def active_model_status(load_check: bool = False) -> Dict[str, Any]:
    import joblib

    def loadable(path: Path) -> dict:
        if not load_check:
            return {"loadable_ok": None, "load_error": None}
        if not path.exists():
            return {"loadable_ok": False, "load_error": "missing"}
        try:
            obj = joblib.load(path)
            return {"loadable_ok": isinstance(obj, dict), "load_error": None if isinstance(obj, dict) else f"expected dict bundle; got {type(obj)}"}
        except Exception as e:
            return {"loadable_ok": False, "load_error": f"{type(e).__name__}: {e}"}

    under30 = resolve_orb_bucket_path(29.99)
    liquid = resolve_orb_bucket_path(30.0)
    rr = resolve_rr_model_path()
    return {
        "strict": {"orb": env_orb_strict_ml(), "rr": env_rr_strict_ml()},
        "orb_under_30": {**file_status(under30), **loadable(under30)},
        "orb_ge_30": {**file_status(liquid), **loadable(liquid)},
        "rr_gold": {**file_status(rr), **loadable(rr)},
    }
