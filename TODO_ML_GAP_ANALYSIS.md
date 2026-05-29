# TODO: ML Scoring for Gap Watchlist Analysis

## What
Wire `entry_now_30m_pm.pkl` into `api_gap_watchlist_analysis()` so each symbol
gets an ML probability score visible in the Analyze Watchlist table.

## Where
- **Backend:** `app.py` → `api_gap_watchlist_analysis()` (~line 4716)
  - For each symbol, compute PM features (same as `api_entry_now` does)
  - Call `_load_entry_now_bundle(pm_path)` + run predict
  - Add `ml_score` field to each row dict
- **Frontend:** `templates/index.html` → analyze table (~line 6237)
  - Add ML column header
  - Render `row.ml_score` with color (green ≥ 0.55, amber 0.40–0.55, red < 0.40)
  - Show "—" if model not loaded or PM bars insufficient

## Model
`models/entry_now_30m_pm.pkl` — pre-market specific, needs ≥ 20 PM bars.
Feature computation lives in `app.py` around line 1850 (`api_entry_now` handler).
Extract that into a shared helper so both endpoints can call it.

## Notes
- Gate on `len(pm_bars) >= 20` before scoring — return `ml_score: null` if insufficient
- Model path: `_ML_STATE["entry_now_pm_path"]`
- Already have `pm_bars` in the analysis loop — just need feature extraction
