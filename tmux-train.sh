#!/bin/bash

tmux new-session -d -s soccer '
set -e

echo "=== Stage 2: Kick (100k) ===" && \
WANDB_MODE=disabled uv run --extra cu124 train Mjlab-Soccer-Kick-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 --agent.max-iterations 100000 && \

KICK_RUN=$(ls -t logs/rsl_rl/g1_soccer/ | head -1) && \
KICK_CKPT=$(basename $(ls -t logs/rsl_rl/g1_soccer/$KICK_RUN/model_*.pt | head -1)) && \

echo "=== Stage 2.5: Moving Ball (20k) ===" && \
WANDB_MODE=disabled uv run --extra cu124 train Mjlab-Soccer-Moving-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run $KICK_RUN \
  --agent.load-checkpoint $KICK_CKPT --agent.max-iterations 20000 && \

echo "=== All stages complete ==="
'

#echo "=== Stage 3a: Teacher (100k) ===" && \
#WANDB_MODE=disabled uv run train Mjlab-Soccer-Teacher-Flat-Unitree-G1 \
#  --env.scene.num-envs 4096 --agent.max-iterations 100000 && \
