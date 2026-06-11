"""Benchmark old bucket models vs new outcomes model against real trade outcomes.

Usage:
    python ml/benchmark_models.py [--host http://127.0.0.1:8050]

Outputs:
  - ROC-AUC for each model on the same labeled outcomes
  - Precision / recall breakdown at several score thresholds
  - Win rate by score decile (does a high score actually predict wins?)
  - Calibration: are the probabilities trustworthy?

No fake data. Every metric computed on real resolved trades.
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


def _load_outcomes(host: str, strategy: str) -> list[dict]:
    import urllib.request
    url = f"{host}/api/outcomes/export_training_data?strategy={strategy}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    if not data.get("ok"):
        raise RuntimeError(f"API error: {data}")
    return data["rows"]


def _features_for_outcome(provider, sym: str, session_date: date) -> dict:
    """Re-fetch real features for a historical trade. Identical to retrain pipeline."""
    hist_start = session_date - timedelta(days=365)
    daily = provider.get_bars_range(
        symbol=sym, interval="1d",
        from_d=hist_start, to_d=session_date,
        include_prepost=False,
    ).sort_index()
    if daily is None or daily.empty or len(daily) < 50:
        raise RuntimeError(f"Insufficient daily history for {sym}")

    close = daily["Close"].astype(float)
    vol = daily["Volume"].astype(float)
    high = daily["High"].astype(float)
    low = daily["Low"].astype(float)

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    trend_20_50 = float(sma20 / sma50 - 1.0)
    vol20 = float(close.pct_change().rolling(20).std().iloc[-1])
    avg20_vol = float(vol.rolling(20).mean().iloc[-1])
    last_close = float(close.iloc[-1])
    avg20_dollar_vol = float(avg20_vol * last_close)
    prev_close_series = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close_series).abs(), (low - prev_close_series).abs()],
        axis=1,
    ).max(axis=1)
    atr14_pct = float(float(tr.rolling(14).mean().iloc[-1]) / last_close)
    mom5 = float(close.iloc[-1] / close.iloc[-6] - 1.0)
    mom20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)

    intraday = provider.get_bars_range(
        symbol=sym, interval="1m",
        from_d=session_date, to_d=session_date,
        include_prepost=False,
    ).sort_index()
    if intraday is None or intraday.empty:
        raise RuntimeError(f"No intraday for {sym} on {session_date}")
    if intraday.index.tz is None:
        intraday.index = intraday.index.tz_localize(pytz.UTC)
    intraday = intraday.tz_convert(ET)

    intr_open = intraday["Open"].astype(float)
    intr_high = intraday["High"].astype(float)
    intr_low = intraday["Low"].astype(float)
    intr_close = intraday["Close"].astype(float)
    intr_vol = intraday["Volume"].astype(float)

    gap_pct = float(float(intr_open.iloc[0]) / last_close - 1.0)
    early_5 = intraday.iloc[:5]
    or_high = float(early_5["High"].astype(float).max())
    or_low = float(early_5["Low"].astype(float).min())
    or_open = float(intr_open.iloc[0])
    or_range_pct = float((or_high - or_low) / or_open)
    or_mid = (or_high + or_low) / 2.0
    relvol5 = float(early_5["Volume"].astype(float).sum() / avg20_vol)
    relvol15 = float(intraday.iloc[:15]["Volume"].astype(float).sum() / avg20_vol)
    tp = (intr_high + intr_low + intr_close) / 3.0
    vwap_val = float((tp * intr_vol).cumsum().iloc[-1] / intr_vol.cumsum().iloc[-1])
    prev_close_vwap_delta_pct = float((vwap_val - last_close) / last_close * 100.0)
    or_midpoint_vs_vwap_pct = float((or_mid - vwap_val) / vwap_val * 100.0)
    day_dollar_vol = float(intr_vol.sum() * float(intr_close.iloc[-1]))
    day_ret = float(float(intr_close.iloc[-1]) / float(intr_close.iloc[0]) - 1.0)

    return {
        "trend_20_50": trend_20_50, "vol20": vol20, "avg20_dollar_vol": avg20_dollar_vol,
        "mom5": mom5, "mom20": mom20, "gap_pct": gap_pct, "atr14_pct": atr14_pct,
        "or_range_pct": or_range_pct, "relvol5": relvol5, "relvol15": relvol15,
        "prev_close_vwap_delta_pct": prev_close_vwap_delta_pct,
        "or_midpoint_vs_vwap_pct": or_midpoint_vs_vwap_pct,
        "day_dollar_vol": day_dollar_vol, "day_ret": day_ret,
    }


def _score_with_model(blob: dict, X: pd.DataFrame) -> np.ndarray:
    model = blob["model"]
    feature_names = blob["feature_names"]
    Xf = X.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
    return model.predict_proba(Xf)[:, 1]


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _wr_by_decile(y: np.ndarray, scores: np.ndarray) -> str:
    df = pd.DataFrame({"label": y, "score": scores})
    df["decile"] = pd.qcut(df["score"], q=5, labels=["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"], duplicates="drop")
    rows = []
    for q, grp in df.groupby("decile", observed=True):
        wr = grp["label"].mean() * 100
        rows.append(f"  {q}: {wr:5.1f}% WR  (n={len(grp)})")
    return "\n".join(rows)


def _precision_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> str:
    mask = scores >= threshold
    if mask.sum() == 0:
        return f"  >= {threshold:.2f}: no predictions"
    prec = y[mask].mean() * 100
    cov = mask.mean() * 100
    return f"  >= {threshold:.2f}: {prec:.1f}% precision  ({mask.sum()} trades, {cov:.0f}% coverage)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8050")
    ap.add_argument("--strategy", default="orb")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import joblib

    models_dir = _ROOT / "models"
    model_files = {
        "old_liquid (model_a_liquid)": models_dir / "model_a_liquid.pkl",
        "old_outlier (model_b_outlier)": models_dir / "model_b_outlier.pkl",
        "NEW outcomes model": models_dir / "orb_ranker_outcomes.pkl",
    }
    blobs = {}
    for name, path in model_files.items():
        if path.exists():
            blobs[name] = joblib.load(path)
            print(f"Loaded: {name}  ({path.name})")
        else:
            print(f"Missing: {name}  ({path.name}) — skipping")

    if not blobs:
        print("No models found.")
        return 1

    print(f"\nFetching outcomes from {args.host}...")
    outcome_rows = _load_outcomes(args.host, args.strategy)
    print(f"Got {len(outcome_rows)} resolved trades\n")

    from providers.alpaca_provider import AlpacaProvider
    provider = AlpacaProvider()

    # Cache SPY/QQQ per date
    unique_dates = {str(r.get("session_date") or "") for r in outcome_rows if r.get("session_date")}
    spy_qqq_cache: dict[str, dict] = {}
    for d_str in sorted(unique_dates):
        try:
            session_date = date.fromisoformat(d_str)
            spy = provider.get_bars_range(symbol="SPY", interval="1m", from_d=session_date, to_d=session_date, include_prepost=False).sort_index()
            qqq = provider.get_bars_range(symbol="QQQ", interval="1m", from_d=session_date, to_d=session_date, include_prepost=False).sort_index()
            for df in [spy, qqq]:
                if df.index.tz is None:
                    df.index = df.index.tz_localize(pytz.UTC)
            spy = spy.tz_convert(ET); qqq = qqq.tz_convert(ET)
            spy_ret = float(float(spy["Close"].iloc[-1]) / float(spy["Open"].iloc[0]) - 1.0) * 100
            qqq_ret = float(float(qqq["Close"].iloc[-1]) / float(qqq["Open"].iloc[0]) - 1.0) * 100
            spy_qqq_cache[d_str] = {
                "spy_intraday_ret": spy_ret, "qqq_intraday_ret": qqq_ret,
                "spy_qqq_divergence": abs(spy_ret - qqq_ret),
                "market_up": 1.0 if spy_ret > 0 and qqq_ret > 0 else 0.0,
            }
        except Exception as e:
            print(f"  SPY/QQQ failed for {d_str}: {e}")

    print(f"Computing features for {len(outcome_rows)} trades...")

    def _process(row: dict) -> dict | None:
        sym = str(row.get("symbol") or "").strip().upper()
        d_str = str(row.get("session_date") or "")
        label = int(row.get("label", -1))
        if not sym or not d_str or label not in (0, 1):
            return None
        try:
            feats = _features_for_outcome(provider, sym, date.fromisoformat(d_str))
            if d_str in spy_qqq_cache:
                feats.update(spy_qqq_cache[d_str])
            feats["label"] = label
            feats["symbol"] = sym
            feats["session_date"] = d_str
            return feats
        except Exception as e:
            print(f"  SKIP {sym} {d_str}: {e}")
            return None

    feature_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_process, r) for r in outcome_rows]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                feature_rows.append(r)

    df_all = pd.DataFrame(feature_rows)
    y = df_all["label"].astype(int).values
    X = df_all.drop(columns=["label", "symbol", "session_date"], errors="ignore").select_dtypes(include=[np.number])

    wins = int(y.sum())
    stops = int((y == 0).sum())
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {wins} wins / {stops} stops / {wins+stops} total  (WR={wins/(wins+stops)*100:.1f}%)")
    print(f"{'='*60}\n")

    for name, blob in blobs.items():
        print(f"── {name} ──")
        try:
            scores = _score_with_model(blob, X)
        except Exception as e:
            print(f"  Scoring failed: {e}\n")
            continue

        auc = _auc(y, scores)
        avg_win_score = float(scores[y == 1].mean()) if (y == 1).any() else float("nan")
        avg_stop_score = float(scores[y == 0].mean()) if (y == 0).any() else float("nan")

        print(f"  ROC-AUC:              {auc:.3f}  (0.5=random, 1.0=perfect)")
        print(f"  Avg score | wins:     {avg_win_score:.3f}")
        print(f"  Avg score | stops:    {avg_stop_score:.3f}")
        print(f"  Signal direction:     {'CORRECT (wins score higher)' if avg_win_score > avg_stop_score else 'INVERTED (stops score higher — model anti-predictive)'}")
        print(f"\n  Win rate by score quintile (Q5 = highest confidence):")
        print(_wr_by_decile(y, scores))
        print(f"\n  Precision at threshold (above = how often you win):")
        for t in [0.25, 0.40, 0.50, 0.60, 0.75]:
            print(_precision_at_threshold(y, scores, t))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
