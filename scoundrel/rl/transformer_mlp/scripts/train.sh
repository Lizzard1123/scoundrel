#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard for this transformer_mlp trainer and run training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"               # scoundrel/rl/transformer_mlp
REPO_ROOT="$(cd "$BASE_DIR/../../.." && pwd)"          # project root

LOGDIR="${LOGDIR:-$BASE_DIR/runs}"
CHECKPOINT="${CHECKPOINT:-$BASE_DIR/checkpoints/ppo_latest.pt}"
PORT="${PORT:-6006}"

# Handle flags (currently only --clear)
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

# Reset "$@" to passthrough args (handle empty array under 'set -u')
if ((${#PASSTHRU[@]})); then
  set -- "${PASSTHRU[@]}"
else
  set --
fi

if [[ "$CLEAR" -eq 1 ]]; then
  echo "Clearing TensorBoard logs at $LOGDIR and checkpoint $CHECKPOINT"
  rm -rf "$LOGDIR"
  rm -f "$CHECKPOINT"
  exit 0
fi

mkdir -p "$LOGDIR" "$BASE_DIR/logs" "$(dirname "$CHECKPOINT")"

# Start TensorBoard if it is not already watching this logdir.
if ! pgrep -f "tensorboard.+--logdir $LOGDIR" >/dev/null 2>&1; then
  echo "Starting TensorBoard on port $PORT (logdir: $LOGDIR)"
  tensorboard --logdir "$LOGDIR" --port "$PORT" > "$BASE_DIR/logs/tensorboard.out" 2>&1 &
else
  echo "TensorBoard already running for logdir $LOGDIR"
fi
echo "TensorBoard: http://localhost:$PORT/"

cd "$REPO_ROOT"
python -m scoundrel.rl.transformer_mlp.transformer_mlp \
  --logdir "$LOGDIR" \
  --checkpoint "$CHECKPOINT" \
  "$@"
