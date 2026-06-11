"""Retrain the ORB ranker using live trade outcomes as ground-truth labels.

Usage:
    python ml/retrain_from_outcomes.py [--host http://127.0.0.1:8050] [--strategy orb] [--dry_run]

Pipeline:
  1. Fetch resolved outcomes (symbol, session_date, label) from /api/outcomes/export_training_data
  2. For each outcome, re-fetch real market data for that specific session date
  3. Compute the full feature set (identical to live scoring features)
  4. Train LightGBM with calibrated probabilities
  5. Cross-validate, report ROC-AUC and feature importances
  6. Save to models/orb_ranker_outcomes.pkl

Features are re-computed from real historical data — not from stored snapshots.
No fallbacks, no fake data.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

ET = pytz.timezone("America/New_York")
_MIN_EARLY_BARS = 5  # minimum 1m bars for OR computation


def _load_outcomes(host: str, strategy: str) -> list[dict]:
    import urllib.request
    url = f"{host}/api/outcomes/export_training_data?strategy={strategy}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    if not data.get("ok"):
        raise RuntimeError(f"API error: {data}")
    return data["rows"]


def _features_for_outcome(provider, sym: str, session_date: date) -> dict[str, float]:
    """Compute ML features for a specific symbol on a specific historical session date."""
    import pandas as pd

    # Daily history ending at session_date
    hist_end = session_date
    hist_start = session_date - timedelta(days=365)
    daily = provider.get_bars_range(
        symbol=sym, interval="1d",
        from_d=hist_start, to_d=hist_end,
        include_prepost=False,
    ).sort_index()
    if daily is None or daily.empty:
        raise RuntimeError(f"No daily history for {sym} on {session_date}")

    close = daily["Close"].astype(float)
    vol = daily["Volume"].astype(float)
    high = daily["High"].astype(float)
    low = daily["Low"].astype(float)

    if len(close) < 50:
        raise RuntimeError(f"Insufficient daily history for {sym} ({len(close)} bars)")

    _sma20 = close.rolling(20).mean().iloc[-1]
    _sma50 = close.rolling(50).mean().iloc[-1]
    if pd.isna(_sma20) or pd.isna(_sma50):
        raise RuntimeError(f"SMA NaN for {sym}")
    trend_20_50 = float(_sma20 / _sma50 - 1.0)

    vol20 = float(close.pct_change().rolling(20).std().iloc[-1])
    if pd.isna(vol20):
        raise RuntimeError(f"vol20 NaN for {sym}")

    _avg20_vol = vol.rolling(20).mean().iloc[-1]
    if pd.isna(_avg20_vol) or _avg20_vol <= 0:
        raise RuntimeError(f"avg20_vol invalid for {sym}")
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
        raise RuntimeError(f"ATR14 NaN for {sym}")
    atr14_pct = float(float(_atr14) / last_close) if last_close > 0 else None
    if atr14_pct is None:
        raise RuntimeError(f"Invalid last_close for {sym}")

    mom5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else None
    mom20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else None
    if mom5 is None or mom20 is None:
        raise RuntimeError(f"Insufficient history for momentum on {sym}")

    # Intraday for the specific session date
    intraday = provider.get_bars_range(
        symbol=sym, interval="1m",
        from_d=session_date, to_d=session_date,
        include_prepost=False,
    ).sort_index()
    if intraday is None or intraday.empty:
        raise RuntimeError(f"No intraday bars for {sym} on {session_date}")
    if intraday.index.tz is None:
        intraday.index = intraday.index.tz_localize(pytz.UTC)
    intraday = intraday.tz_convert(ET)

    intr_open = intraday["Open"].astype(float) if "Open" in intraday.columns else intraday["Close"].astype(float)
    intr_high = intraday["High"].astype(float) if "High" in intraday.columns else intraday["Close"].astype(float)
    intr_low = intraday["Low"].astype(float) if "Low" in intraday.columns else intraday["Close"].astype(float)
    intr_close = intraday["Close"].astype(float)
    intr_vol = intraday["Volume"].astype(float)

    today_open = float(intr_open.iloc[0])
    prev_close_val = last_close
    if prev_close_val <= 0:
        raise RuntimeError(f"Invalid prev_close for {sym}")
    if today_open <= 0:
        raise RuntimeError(f"Invalid today_open for {sym}")
    gap_pct = float(today_open / prev_close_val - 1.0)

    early_5 = intraday.iloc[:_MIN_EARLY_BARS]
    if len(early_5) < _MIN_EARLY_BARS:
        raise RuntimeError(f"Fewer than {_MIN_EARLY_BARS} early bars for {sym} on {session_date}")
    or_high = float(early_5["High"].astype(float).max()) if "High" in early_5.columns else float(early_5["Close"].astype(float).max())
    or_low = float(early_5["Low"].astype(float).min()) if "Low" in early_5.columns else float(early_5["Close"].astype(float).min())
    or_open = float(intr_open.iloc[0])
    if or_open <= 0:
        raise RuntimeError(f"Invalid OR open for {sym}")
    or_range_pct = float((or_high - or_low) / or_open)
    or_mid = (or_high + or_low) / 2.0

    relvol5 = float(early_5["Volume"].astype(float).sum() / avg20_vol)
    early_15 = intraday.iloc[:15]
    relvol15 = float(early_15["Volume"].astype(float).sum() / avg20_vol)

    # VWAP
    if all(c in intraday.columns for c in ("High", "Low", "Close", "Volume")):
        tp = (intr_high + intr_low + intr_close) / 3.0
        cum_vol = intr_vol.cumsum()
        if float(cum_vol.iloc[-1]) <= 0:
            raise RuntimeError(f"Zero cumulative volume for {sym} on {session_date}")
        vwap_val = float((tp * intr_vol).cumsum().iloc[-1] / cum_vol.iloc[-1])
        if vwap_val <= 0:
            raise RuntimeError(f"Invalid VWAP for {sym}")
    else:
        raise RuntimeError(f"Missing OHLCV for VWAP on {sym}")

    prev_close_vwap_delta_pct = float((vwap_val - prev_close_val) / prev_close_val * 100.0)
    or_midpoint_vs_vwap_pct = float((or_mid - vwap_val) / vwap_val * 100.0)

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
        "prev_close_vwap_delta_pct": float(prev_close_vwap_delta_pct),
        "or_midpoint_vs_vwap_pct": float(or_midpoint_vs_vwap_pct),
        "day_dollar_vol": float(float(intr_vol.sum()) * float(intr_close.iloc[-1])),
        "day_ret": float((float(intr_close.iloc[-1]) / float(intr_close.iloc[0]) - 1.0) if float(intr_close.iloc[0]) > 0 else 0.0),
    }


def _fetch_spy_qqq_for_date(provider, session_date: date) -> dict[str, float]:
    """Fetch SPY + QQQ features for a specific historical session date."""
    def _ret(sym: str) -> float:
        bars = provider.get_bars_range(
            symbol=sym, interval="1m",
            from_d=session_date, to_d=session_date,
            include_prepost=False,
        ).sort_index()
        if bars is None or bars.empty:
            raise RuntimeError(f"{sym} bars empty for {session_date}")
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize(pytz.UTC)
        bars = bars.tz_convert(ET)
        open_col = "Open" if "Open" in bars.columns else "open"
        close_col = "Close" if "Close" in bars.columns else "close"
        first_open = float(bars[open_col].iloc[0])
        last_close = float(bars[close_col].iloc[-1])
        if first_open <= 0:
            raise RuntimeError(f"{sym} invalid open for {session_date}")
        return float((last_close / first_open - 1.0) * 100.0)

    spy_ret = _ret("SPY")
    qqq_ret = _ret("QQQ")
    return {
        "spy_intraday_ret": spy_ret,
        "qqq_intraday_ret": qqq_ret,
        "spy_qqq_divergence": float(abs(spy_ret - qqq_ret)),
        "market_up": 1.0 if spy_ret > 0 and qqq_ret > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8050")
    ap.add_argument("--strategy", default="orb")
    ap.add_argument("--min_samples", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default=str(_ROOT / "models" / "orb_ranker_outcomes.pkl"))
    args = ap.parse_args()

    print(f"[retrain] Fetching {args.strategy} outcomes from {args.host}...")
    outcome_rows = _load_outcomes(args.host, args.strategy)
    print(f"[retrain] Got {len(outcome_rows)} resolved trades")

    if len(outcome_rows) < args.min_samples:
        print(f"[retrain] Only {len(outcome_rows)} samples — need {args.min_samples}. Aborting.")
        return 1

    from providers.alpaca_provider import AlpacaProvider
    provider = AlpacaProvider()

    # Cache SPY/QQQ per unique session date
    print("[retrain] Fetching SPY/QQQ context per session date...")
    unique_dates: set[str] = {str(r.get("session_date") or "") for r in outcome_rows if r.get("session_date")}
    spy_qqq_cache: dict[str, dict] = {}
    for d_str in sorted(unique_dates):
        try:
            session_date = date.fromisoformat(d_str)
            spy_qqq_cache[d_str] = _fetch_spy_qqq_for_date(provider, session_date)
            print(f"  {d_str} SPY={spy_qqq_cache[d_str]['spy_intraday_ret']:+.2f}% QQQ={spy_qqq_cache[d_str]['qqq_intraday_ret']:+.2f}%")
        except Exception as e:
            print(f"  {d_str} SPY/QQQ fetch failed: {e} — skipping trades from this date")

    # Compute features per outcome in parallel
    print(f"[retrain] Computing features for {len(outcome_rows)} outcomes ({args.workers} workers)...")

    def _process(row: dict) -> dict | None:
        sym = str(row.get("symbol") or "").strip().upper()
        d_str = str(row.get("session_date") or "")
        label = int(row.get("label", -1))
        if not sym or not d_str or label not in (0, 1):
            return None
        if d_str not in spy_qqq_cache:
            return None  # no market context for this date — skip
        try:
            session_date = date.fromisoformat(d_str)
            feats = _features_for_outcome(provider, sym, session_date)
            feats.update(spy_qqq_cache[d_str])
            feats["label"] = label
            feats["symbol"] = sym
            feats["session_date"] = d_str
            return feats
        except Exception as e:
            print(f"  SKIP {sym} {d_str}: {e}")
            return None

    feature_rows = []
    skip_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_process, r): r for r in outcome_rows}
        for fut in as_completed(futs):
            result = fut.result()
            if result is not None:
                feature_rows.append(result)
            else:
                skip_count += 1

    print(f"[retrain] Feature rows: {len(feature_rows)}  skipped: {skip_count}")
    if len(feature_rows) < args.min_samples:
        print(f"[retrain] Not enough usable rows ({len(feature_rows)} < {args.min_samples}). Aborting.")
        return 1

    df = pd.DataFrame(feature_rows)
    y = df["label"].astype(int)
    X = df.drop(columns=["label", "symbol", "session_date"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    wins = int(y.sum())
    stops = int((y == 0).sum())
    print(f"[retrain] Features ({len(X.columns)}): {list(X.columns)}")
    print(f"[retrain] Label distribution: wins={wins}  stops={stops}  WR={wins/(wins+stops)*100:.1f}%")

    if wins < 5 or stops < 5:
        print("[retrain] Not enough of each class for CV. Aborting.")
        return 1

    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=max(3, stops // 3),
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    n_splits = min(5, wins, stops)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(lgbm, X, y, cv=cv, scoring="roc_auc")
    print(f"[retrain] CV ROC-AUC ({n_splits}-fold): {scores.mean():.3f} ± {scores.std():.3f}")

    if scores.mean() < 0.52:
        print(f"[retrain] WARNING: ROC-AUC {scores.mean():.3f} is near random. "
              f"With {stops} stops out of {wins+stops} trades, the ORB WR is high but stop patterns may not be learnable yet.")

    calibrated = CalibratedClassifierCV(lgbm, cv=cv, method="sigmoid")
    calibrated.fit(X, y)

    try:
        base = calibrated.calibrated_classifiers_[0].estimator
        importances = sorted(zip(X.columns, base.feature_importances_), key=lambda x: -x[1])
        print("[retrain] Top features:")
        for feat, imp in importances[:15]:
            print(f"  {feat:45s} {imp:.4f}")
    except Exception:
        pass

    if args.dry_run:
        print("[retrain] Dry run — model not saved.")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    blob = {
        "model": calibrated,
        "feature_names": list(X.columns),
        "strategy": args.strategy,
        "n_samples": len(feature_rows),
        "wins": wins,
        "stops": stops,
        "cv_roc_auc_mean": float(scores.mean()),
        "cv_roc_auc_std": float(scores.std()),
        "source": "trade_outcomes_with_market_data",
    }
    joblib.dump(blob, out_path)
    print(f"[retrain] Model saved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
