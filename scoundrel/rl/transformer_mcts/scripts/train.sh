#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard for this transformer_mcts trainer and run training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"               # scoundrel/rl/transformer_mcts
REPO_ROOT="$(cd "$BASE_DIR/../../.." && pwd)"          # project root

LOGDIR="${LOGDIR:-$BASE_DIR/runs}"
CHECKPOINT="${CHECKPOINT:-$BASE_DIR/checkpoints/ppo_latest.pt}"
PORT="${PORT:-6006}"

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
python -m scoundrel.rl.transformer_mcts.transformer_mcts \
  --logdir "$LOGDIR" \
  --checkpoint "$CHECKPOINT" \
  "$@"
