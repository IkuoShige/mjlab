#!/bin/bash
# V1: LVDRS-style K1 kick task — virtual perception + mirror symmetry loss
# (single-critic PPO; AMP discriminator infra wired but not yet plumbed
# into the runner).
#
# Usage: ./tmux-kick-v1.sh
# Attach:  tmux attach -t kick-v1
# Stop:    tmux kill-session -t kick-v1

set -e
cd "$(dirname "$0")"

SESSION=kick-v1
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-20000}

tmux new-session -d -s "$SESSION" "
set -e
echo '=== K1 Kick V1 (mirror symmetry, virtual perception, MLP PPO) ==='
echo 'num_envs=$NUM_ENVS  max_iterations=$MAX_ITER'
WANDB_MODE=disabled MJLAB_DISABLE_CUDNN=1 uv run --extra cu124 train \
  Mjlab-Kick-Flat-Booster-K1 \
  --env.scene.num-envs $NUM_ENVS \
  --agent.max-iterations $MAX_ITER
echo '=== finished ==='
exec bash
"

echo "Launched tmux session: $SESSION"
echo "Attach:  tmux attach -t $SESSION"
echo "Stop:    tmux kill-session -t $SESSION"
