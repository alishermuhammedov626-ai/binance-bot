#!/usr/bin/env bash
# One-command backtest run for an unattended server.
#
#   ./scripts/server_backtest.sh BTCUSDT 365
#
# Downloads (resumably) if the CSV is missing or stale, runs the full
# section-91 pipeline, and writes every artifact into runs/<timestamp>/ so
# separate runs never overwrite each other:
#
#   log.txt        full console output
#   metrics.json   machine-readable results
#   smcbot.db      SQLite journal: trades, signals, models
#   config.json    the exact config the run used
#
# Exit code: 0 = the pipeline's production checks all passed, 2 = they did not
# (a normal outcome, not an error), 1 = the run itself failed.
set -euo pipefail

SYMBOL="${1:-BTCUSDT}"
DAYS="${2:-365}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DATA="data/${SYMBOL}_1m.csv"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${SYMBOL}-${STAMP}"
mkdir -p "$RUN_DIR" data

exec > >(tee -a "$RUN_DIR/log.txt") 2>&1

echo "=== smcbot server run ==============================================="
echo "symbol   : $SYMBOL"
echo "history  : $DAYS days"
echo "run dir  : $RUN_DIR"
echo "python   : $($PYTHON -V 2>&1)"
echo "started  : $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
echo

"$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" || {
    echo "ERROR: Python 3.9+ required"; exit 1; }

# --- data ------------------------------------------------------------------
# --resume makes an interrupted download continue instead of starting over,
# which matters when a year of M1 takes a few hundred requests.
echo "--- fetching $SYMBOL M1 data ---"
if ! "$PYTHON" -m smcbot fetch --symbol "$SYMBOL" --days "$DAYS" \
        --out "$DATA" --resume; then
    if [ -s "$DATA" ]; then
        echo "WARNING: fetch incomplete, continuing with what is on disk"
    else
        echo "ERROR: no data and the download failed"; exit 1
    fi
fi

ROWS=$(( $(wc -l < "$DATA") - 1 ))
echo "data: $ROWS candles in $DATA"
if [ "$ROWS" -lt 100000 ]; then
    echo "WARNING: under ~70 days of M1 data. The spec asks for 500-3000"
    echo "         trades, which needs roughly 1-3 years. Results from a"
    echo "         short window are noise, not evidence."
fi

# --- config snapshot -------------------------------------------------------
"$PYTHON" -m smcbot config --out "$RUN_DIR/config.json" >/dev/null
echo

# --- pipeline --------------------------------------------------------------
echo "--- pipeline: in-sample -> out-of-sample -> walk-forward -> Monte Carlo ---"
START=$(date +%s)
set +e
"$PYTHON" -m smcbot pipeline --csv "$DATA" \
    --config "$RUN_DIR/config.json" \
    --journal "$RUN_DIR/smcbot.db" \
    --mc-runs 2000
STATUS=$?
set -e
echo
echo "elapsed: $(( ($(date +%s) - START) / 60 )) min"

# --- machine-readable metrics ---------------------------------------------
"$PYTHON" -m smcbot backtest --csv "$DATA" --config "$RUN_DIR/config.json" \
    --json "$RUN_DIR/metrics.json" >/dev/null

echo "finished : $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
echo "artifacts: $RUN_DIR"
case "$STATUS" in
    0) echo "verdict  : all production checks PASSED" ;;
    2) echo "verdict  : production checks FAILED (expected until there is a real edge)" ;;
    *) echo "verdict  : run ERROR (exit $STATUS)" ;;
esac
exit "$STATUS"
