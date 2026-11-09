#!/usr/bin/env bash
# Reproduce the tinker GRPO-vs-OPD replication at 9B scale (Appendix D.9, Figure 20).
#
# Both arms are launched from the same entrypoint and start from the same public
# rank-128 SFT checkpoint of Qwen3.5-9B-Base (see README.md). Every shared knob
# (lr 1e-4, 64 prompts x 4 rollouts, 16384-token cap, DeepMath seed 0,
# temperature 1.0, AIME'24 mean@8 eval every 5 steps) is the default in
# tinker_cookbook/recipes/distillation/opd_vs_grpo.py, so the two commands below
# differ only in `method`.
#
# Our runs were launched for 20 steps and later resumed to 40 (stage 1 + stage 2
# below). Optimizer state carries over on resume and batches are indexed by step,
# so the result is equivalent to one 40-step run; `RUN_MODE=single` does that.
set -euo pipefail

: "${TINKER_API_KEY:?export TINKER_API_KEY first}"
# Metrics go to W&B by default; export WANDB_API_KEY or append wandb_project=None.

cd "$(dirname "$0")/tinker-cookbook"
# Kept apart from ../runs, which holds the logs of our original runs.
RUNS="${RUNS:-$PWD/../reproduced_runs}"
RUN_MODE="${RUN_MODE:-two_stage}"

# The public SFT checkpoint carries no renderer metadata (the lookup 404s), so the
# renderer is pinned to qwen3_5, the format the SFT checkpoint was trained with.
COMMON=(renderer_name=qwen3_5)

launch() {
    local method=$1 arm=$2
    local log_path="$RUNS/$arm"
    if [ "$RUN_MODE" = single ]; then
        uv run python -m tinker_cookbook.recipes.distillation.opd_vs_grpo \
            method="$method" "${COMMON[@]}" num_train_steps=40 log_path="$log_path"
    else
        # Stage 1: steps 0-20 (the original launch used the default num_train_steps=20).
        uv run python -m tinker_cookbook.recipes.distillation.opd_vs_grpo \
            method="$method" "${COMMON[@]}" num_train_steps=20 log_path="$log_path"
        # Stage 2: resume from the step-21 checkpoint (weights + optimizer) to step 40.
        uv run python -m tinker_cookbook.recipes.distillation.opd_vs_grpo \
            method="$method" "${COMMON[@]}" num_train_steps=40 log_path="$log_path" \
            behavior_if_log_dir_exists=resume
    fi
}

launch grpo grpo &
launch opd opd_matched &
wait
