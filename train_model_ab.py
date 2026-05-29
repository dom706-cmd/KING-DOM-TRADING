from __future__ import annotations

"""Train the lightweight ORB models used by the scanner.

These are the historical "Model A" and "Model B" bundles:
  - models/model_a_liquid.pkl  (mode=liquid)
  - models/model_b_outlier.pkl (mode=outlier)

They intentionally use ONLY the 5 daily-derived features:
  trend_20_50, vol20, avg20_dollar_vol, mom5, mom20

Label definition is IDENTICAL to train_ranker.py:
  y=1 if +2R target is hit before stop after an OR-high breakout,
  within `label_minutes` minutes after the opening 5m bar.

Tenants:
  - Real provider data only (Massive). No placeholders.
  - Provider failures are recorded; training fails only on real insufficiency.
  - Produces a real sklearn model persisted as a .pkl bundle via joblib.
"""

import argparse
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from providers.alpaca_provider import AlpacaProvider


_ET = ZoneInfo("America/New_York")


def _with_retries(fn, *, attempts: int, failures: List[dict], symbol: str, stage: str):
    last = None
    for i in range(1, max(1, int(attempts)) + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            failures.append({"symbol": symbol, "stage": stage, "attempt": i, "error": f"{type(e).__name__}: {e}"})
    raise last  # type: ignore[misc]


def _is_common_equity(sym: str) -> bool:
    s = sym.strip().upper()
    if "$" in s:
        return False
    if re.search(r"\.(W|R)$", s):
        return False
    return True


def _load_symbols(args) -> List[str]:
    if args.universe:
        uni = args.universe.strip().lower()
        if uni == "file":
            p = Path("symbols.txt")
            if not p.exists():
                raise SystemExit("symbols.txt not found for universe=file")
            syms = [
                ln.strip().upper()
                for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        else:
            from universe.nasdaq_symbols import get_nasdaq_symbols

            syms = get_nasdaq_symbols(
                include_non_common=bool(args.include_non_common),
                cache_dir=str(getattr(args, "cache_dir", "cache")),
                ttl_seconds=int(getattr(args, "ttl_seconds", 86400)),
            )

        if not args.include_non_common:
            syms = [s for s in syms if _is_common_equity(s)]

        start = int(args.offset)
        end = start + int(args.max_symbols)
        return syms[start:end]

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not args.include_non_common:
            syms = [s for s in syms if _is_common_equity(s)]
        return syms

    if args.symbols_file:
        p = Path(args.symbols_file)
        df = pd.read_csv(p)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        syms = [str(x).strip().upper() for x in df[col].dropna().tolist()]
        if not args.include_non_common:
            syms = [s for s in syms if _is_common_equity(s)]
        return syms

    raise SystemExit("Provide --symbols or --symbols_file or --universe")


def _regular_session_day(df_5m: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    if df_5m is None or df_5m.empty:
        return pd.DataFrame()
    df = df_5m
    if getattr(df.index, "tz", None) is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df_et = df.tz_convert(_ET)

    d = pd.Timestamp(day)
    d_et = d.tz_localize(_ET) if d.tzinfo is None else d.tz_convert(_ET)
    start = d_et.normalize() + pd.Timedelta(hours=9, minutes=30)
    end = d_et.normalize() + pd.Timedelta(hours=16, minutes=0)
    out = df_et[(df_et.index >= start) & (df_et.index < end)]
    return out.tz_convert("UTC")


def _first_bar_or(df_5m: pd.DataFrame, day: pd.Timestamp) -> Tuple[float, float]:
    day_df = _regular_session_day(df_5m, day)
    if day_df is None or day_df.empty:
        raise RuntimeError("No regular-session bars for day")
    first = day_df.iloc[0]
    return float(first["High"]), float(first["Low"])


def _label_day(df_5m: pd.DataFrame, day: pd.Timestamp, *, label_minutes: int) -> int:
    day_df = _regular_session_day(df_5m, day)
    if day_df is None or day_df.empty or len(day_df) < 3:
        raise RuntimeError("Not enough intraday bars for day")

    or_high, or_low = _first_bar_or(df_5m, day)
    entry = or_high
    stop = or_low
    risk = entry - stop
    if risk <= 0:
        raise RuntimeError("Invalid OR risk")
    target = entry + 2.0 * risk

    in_pos = False
    max_bars = max(1, int(label_minutes) // 5)
    window = day_df.iloc[1 : 1 + max_bars]
    for _, row in window.iterrows():
        hi = float(row["High"])
        lo = float(row["Low"])

        if not in_pos:
            if hi >= entry:
                in_pos = True
            else:
                continue
        if lo <= stop:
            return 0
        if hi >= target:
            return 1
    return 0


def _features_day_light(daily: pd.DataFrame, day: pd.Timestamp) -> Dict[str, float]:
    d = daily.copy().sort_index()
    d = d[d.index.normalize() < day.normalize()]
    if len(d) < 60:
        raise RuntimeError("Not enough daily history")

    close = d["Close"].astype(float)
    vol = d["Volume"].astype(float)

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    trend_20_50 = 1.0 if sma20 > sma50 else 0.0

    ret = close.pct_change()
    vol20 = float(ret.rolling(20).std().iloc[-1] or 0.0)

    avg20_vol = float(vol.rolling(20).mean().iloc[-1] or 0.0)
    last_close = float(close.iloc[-1])
    avg20_dollar_vol = float(avg20_vol * last_close)

    mom5 = float(close.iloc[-1] / close.iloc[-6] - 1.0)
    mom20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)

    return {
        "trend_20_50": float(trend_20_50),
        "vol20": float(vol20),
        "avg20_dollar_vol": float(avg20_dollar_vol),
        "mom5": float(mom5),
        "mom20": float(mom20),
    }


def _fetch_intraday_5m_multi_year(
    provider: AlpacaProvider,
    sym: str,
    *,
    years: float,
    chunk_days: int,
    timeout_s: int,
    retries: int,
    failures: List[dict],
) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=int(float(years) * 366) + 10)

    dfs: list[pd.DataFrame] = []
    cur = start
    cd = max(3, int(chunk_days))

    while cur < end:
        nxt = min(end, cur + timedelta(days=cd))

        def _do_5m():
            return provider.get_bars_range(
                symbol=sym,
                interval="5m",
                from_d=cur,
                to_d=nxt,
                include_prepost=False,
                timeout_s=int(timeout_s),
            )

        try:
            df = _with_retries(_do_5m, attempts=int(retries), failures=failures, symbol=sym, stage="fetch_5m_chunk")
        except Exception:
            df = pd.DataFrame()
        if df is not None and not df.empty:
            dfs.append(df)
        cur = nxt

    if not dfs:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = pd.concat(dfs).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["liquid", "outlier"], default="liquid")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--universe", default="", help="nasdaq or file (symbols.txt)")
    ap.add_argument("--symbols_file", default="")
    ap.add_argument("--max_symbols", type=int, default=2500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--include_non_common", action="store_true", default=False)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--ttl_seconds", type=int, default=86400)

    ap.add_argument("--intraday_years", type=float, default=2.0)
    ap.add_argument("--label_minutes", type=int, default=60)
    ap.add_argument("--chunk_days", type=int, default=7)
    ap.add_argument("--provider_timeout_s", type=int, default=25)
    ap.add_argument("--provider_retries", type=int, default=3)

    ap.add_argument("--price_max", type=float, default=30.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    mode = (args.mode or "liquid").strip().lower()
    if mode == "outlier" and int(args.label_minutes) <= 0:
        args.label_minutes = 90
    if mode == "liquid" and int(args.label_minutes) <= 0:
        args.label_minutes = 60

    out_path = Path(args.out) if args.out else Path("models") / ("model_a_liquid.pkl" if mode == "liquid" else "model_b_outlier.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    symbols = _load_symbols(args)
    provider = AlpacaProvider()

    failures: List[dict] = []
    timeout_s = int(args.provider_timeout_s)
    retries = int(args.provider_retries)

    X_rows: List[Dict[str, float]] = []
    y: List[int] = []

    day_rows_attempted = 0
    day_rows_added = 0
    day_rows_skipped = 0

    for i, sym in enumerate(symbols, start=1):
        if i == 1 or (i % 50) == 0:
            print(f"[{i}/{len(symbols)}] fetch+label {sym} (rows={len(X_rows)})", flush=True)

        try:
            df5 = _fetch_intraday_5m_multi_year(
                provider,
                sym,
                years=float(args.intraday_years),
                chunk_days=int(args.chunk_days),
                timeout_s=int(timeout_s),
                retries=int(retries),
                failures=failures,
            ).sort_index()
        except Exception as e:
            failures.append({"symbol": sym, "stage": "fetch_5m", "error": f"{type(e).__name__}: {e}"})
            continue

        if df5 is None or df5.empty:
            failures.append({"symbol": sym, "stage": "empty_5m", "error": "empty dataframe"})
            continue

        try:
            daily = provider.get_daily_history(sym, period="5y").sort_index()
        except Exception as e:
            failures.append({"symbol": sym, "stage": "fetch_daily", "error": f"{type(e).__name__}: {e}"})
            continue

        if daily is None or daily.empty:
            failures.append({"symbol": sym, "stage": "empty_daily", "error": "empty dataframe"})
            continue

        days = sorted({ts.normalize() for ts in df5.index})
        if len(days) > 2000:
            days = days[-2000:]

        for day in days:
            day_rows_attempted += 1
            try:
                day_df = df5.loc[day.strftime("%Y-%m-%d")]
                if day_df is None or day_df.empty:
                    raise RuntimeError("No bars for day")
                first = day_df.iloc[0]
                px_open = float(first["Open"])
                if px_open <= 0:
                    raise ValueError("filtered_bad_open")
                if mode == "liquid" and px_open >= float(args.price_max):
                    raise ValueError("filtered_over_price_max")
                if mode == "outlier":
                    if px_open >= 20.0:
                        raise ValueError("filtered_not_under_20")
                    day_rth = _regular_session_day(df5, day)
                    if day_rth is None or day_rth.empty or len(day_rth) < 3:
                        raise RuntimeError("Not enough intraday bars for day")
                    first_rth = day_rth.iloc[0]
                    or_high = float(first_rth["High"])
                    or_low = float(first_rth["Low"])
                    or_mid = (or_high + or_low) / 2.0
                    or_range_pct = (or_high - or_low) / max(1e-9, or_mid) * 100.0
                    if or_range_pct < 2.0:
                        raise ValueError("filtered_not_outlier")

                feats = _features_day_light(daily, day)
                label = _label_day(df5, day, label_minutes=int(args.label_minutes))
            except Exception as e:
                day_rows_skipped += 1
                failures.append({"symbol": sym, "stage": "feature_or_label", "error": f"{type(e).__name__}: {e}"})
                continue

            X_rows.append(feats)
            y.append(int(label))
            day_rows_added += 1

    print("\n=== Training data summary ===")
    print(f"mode: {mode}")
    print(f"symbols attempted: {len(symbols)}")
    print(f"day rows attempted: {day_rows_attempted}")
    print(f"day rows added: {day_rows_added}")
    print(f"day rows skipped: {day_rows_skipped}")
    print(f"failures recorded: {len(failures)}")
    if failures:
        print("failures (first 10):")
        for f in failures[:10]:
            print("  ", f)

    MIN_ROWS = 500
    if len(X_rows) < MIN_ROWS:
        raise RuntimeError(
            f"Too few training samples: {len(X_rows)} rows collected (<{MIN_ROWS}). "
            "This is a real failure: provider data coverage was insufficient for the chosen universe/limits."
        )
    if len(set(y)) < 2:
        raise RuntimeError(
            f"Training labels collapsed to a single class (unique={set(y)}). "
            "This is a real failure: adjust labeling thresholds or expand data."
        )

    X = pd.DataFrame(X_rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_names = list(X.columns)
    y_arr = np.asarray(y, dtype=int)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y_arr)
    proba = model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y_arr, proba))
    print(f"AUC(in-sample)={auc:.4f}  (light AB model)")

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "mode": mode,
        "label_minutes": int(args.label_minutes),
        "intraday_years": float(args.intraday_years),
        "price_max": float(args.price_max),
        "rows": int(len(X)),
        "pos_rate": float(np.mean(y_arr)),
        "auc_insample": float(auc),
    }
    joblib.dump(bundle, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
