#!/usr/bin/env bash
set -euo pipefail

# MCTS Training Script for Scoundrel
# Runs MCTS agent and collects statistics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"               # scoundrel/rl/mcts
REPO_ROOT="$(cd "$BASE_DIR/../../.." && pwd)"          # project root

LOGDIR="${LOGDIR:-$BASE_DIR/logs}"

# Handle flags
CLEAR=0
PASSTHRU=()
for arg in "$@"; do
  case "$arg" in
    --clear)
      CLEAR=1
      ;;
    *)
      PASSTHRU+=("$arg")
      ;;
  esac
done

# Reset "$@" to passthrough args
if ((${#PASSTHRU[@]})); then
  set -- "${PASSTHRU[@]}"
else
  set --
fi

if [[ "$CLEAR" -eq 1 ]]; then
  echo "Clearing MCTS logs at $LOGDIR"
  rm -rf "$LOGDIR"
  exit 0
fi

mkdir -p "$LOGDIR"

echo "=== MCTS Training for Scoundrel ==="
echo "Logs directory: $LOGDIR"
echo ""

cd "$REPO_ROOT"
python -m scoundrel.rl.mcts.mcts "$@"



