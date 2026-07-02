#!/bin/zsh
# Weekly retrain of the Kingdom "outcomes" ML models from the R-multiple edge tracker.
#
# For each strategy it: backs up the current model, retrains + saves from the live
# trade_outcomes DB (runtime/runtime_state.db via --local), and appends a CV-AUC trend row
# to reports/ml_retrain_trend.csv so the orb (and others) accuracy trend can be watched
# over time. The per-strategy usability gate in ml/orb_model_service.py then auto-promotes
# or demotes each model on the next scan (e.g. orb takes over once its CV-AUC clears 0.58).
#
# Scheduled weekly via ~/Library/LaunchAgents/com.kingdom.weekly-retrain.plist.
# Run manually any time:  scripts/weekly_retrain.sh
set -uo pipefail

ROOT="/Users/dominicaleandri/KingDom"
cd "$ROOT" || { echo "[weekly_retrain] cannot cd to $ROOT" >&2; exit 1; }

# Alpaca creds: a login shell (zsh -lc) should already provide them the same way the app does.
# If not, drop ALPACA_API_KEY / ALPACA_SECRET_KEY into this optional file and it is sourced here.
[ -f "$HOME/.config/kingdom/retrain.env" ] && source "$HOME/.config/kingdom/retrain.env"

PY="$ROOT/.venv/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
DAY="$(date +%Y%m%d)"
REPORTS="$ROOT/reports"
mkdir -p "$REPORTS"
LOG="$REPORTS/weekly_retrain_${STAMP}.log"
TREND="$REPORTS/ml_retrain_trend.csv"

echo "[weekly_retrain] start $STAMP" | tee -a "$LOG"

# Back up current outcomes models before overwriting. Keyed on the full timestamp so a
# second run on the same day cannot clobber the earlier (pre-retrain) backup.
BACKUP="$ROOT/models/backup_${STAMP}"
mkdir -p "$BACKUP"
cp -p "$ROOT"/models/*_outcomes.pkl "$BACKUP"/ 2>/dev/null && \
  echo "[weekly_retrain] backed up current models -> $BACKUP" | tee -a "$LOG"

# CSV header on first run.
[ -f "$TREND" ] || echo "date,strategy,cv_roc_auc_mean,cv_std,n_samples,wins,stops,saved" > "$TREND"

STRATS=(orb parabolic eod_momentum atr_expansion)
for strat in "${STRATS[@]}"; do
  out="models/${strat}_ranker_outcomes.pkl"
  echo "===== $strat @ $STAMP =====" >> "$LOG"
  run="$("$PY" ml/retrain_from_outcomes.py --local --strategy "$strat" --workers 8 --out "$out" 2>&1)"
  echo "$run" >> "$LOG"

  auc="$(echo "$run"   | grep -oE 'CV ROC-AUC \([0-9]+-fold\): [0-9.]+' | grep -oE '[0-9.]+$' | tail -1)"
  std="$(echo "$run"   | grep -oE '± [0-9.]+'          | grep -oE '[0-9.]+' | tail -1)"
  n="$(echo "$run"     | grep -oE 'Feature rows: [0-9]+' | grep -oE '[0-9]+'  | tail -1)"
  wins="$(echo "$run"  | grep -oE 'wins=[0-9]+'        | grep -oE '[0-9]+'  | tail -1)"
  stops="$(echo "$run" | grep -oE 'stops=[0-9]+'       | grep -oE '[0-9]+'  | tail -1)"
  if echo "$run" | grep -q 'Model saved'; then saved=yes; else saved=no; fi

  echo "${DAY},${strat},${auc:-NA},${std:-NA},${n:-NA},${wins:-NA},${stops:-NA},${saved}" >> "$TREND"
  echo "[weekly_retrain] $strat: auc=${auc:-NA} n=${n:-NA} saved=${saved}" | tee -a "$LOG"
done

echo "[weekly_retrain] done $STAMP -> $TREND" | tee -a "$LOG"
