# GRPO vs. OPD on tinker at 9B scale

This is the code and configuration for the third framework replication in
Appendix D.9 ("tinker replication at 9B scale", Figure 20). It is the public
[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) at
commit `ab78b73` (upstream "Release 2026-08-13" plus one renderer commit), with
one new launcher and a few small additions. The training logs and metrics for
both runs are included in `runs/`. The large per-step rollout transcripts and
evaluation generations (about 2 GB) are left out.

## What was run

Both arms start from the same checkpoint: Thinking Machines' public **rank-128
LoRA SFT checkpoint of Qwen3.5-9B-Base**, trained on OpenThoughts3. It is listed
in the cookbook's `tinker_cookbook/recipes/distillation/README.md`:

```
tinker://b89f6676-49c4-53ed-9788-2e0c81a9eabf:train:0/weights/final
```

This checkpoint is the default `load_checkpoint_path` in
`opd_vs_grpo.py`, and both training logs report loading it at step 0. Before
any update, both arms score the same AIME'24 mean@8 of 0.500 (120 of 240).

The last column shows Thinking Machines' official recipe for reproducing
on-policy distillation on this student: the reasoning command in the
[distillation tutorial](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/),
plus the defaults of `on_policy_distillation.py` for any setting the command
leaves unset. ✓ means our setting matches it, and ✗ means it differs. The
tutorial has no GRPO arm, so the comparison is with our OPD run.

| | GRPO | OPD | Official tutorial (OPD) |
|---|---|---|---|
| Student / init | Qwen3.5-9B-Base + rank-128 SFT checkpoint above | same | same checkpoint ✓ |
| LoRA rank | 128 (inherited from the checkpoint) | same | 128 ✓ |
| Training signal | `MathEnv` correctness reward (+0.1 format term), group-centered advantages | zero environment reward; advantage = −(log p_student − log p_teacher) per token (reverse KL, `kl_penalty_coef=1.0`, `kl_discount_factor=0`) | reverse KL, `kl_penalty_coef=1.0`, `kl_discount_factor=0` ✓ |
| Teacher | none | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-9B` ✓ |
| Data | DeepMath-103K, shuffled with seed 0 | same dataset, same order, same prompt text (`opd_dataset=matched`) | DeepMath-103K ✓; bare question, prompt cut at 1 024 tokens, unshuffled ✗ (see below) |
| Batch | 64 prompts × 4 rollouts | same | 512 prompts × 4 rollouts ✗ |
| Learning rate | 1e-4, constant, Adam (β₁=0.9, β₂=0.95) | same | 1e-4 ✓ |
| Rollout cap / temperature | 16 384 tokens / 1.0 | same | 16 384 / 1.0 ✓ |
| Loss | `importance_sampling` | same | `importance_sampling` ✓ |
| Steps | 40 | same | 200 ✗ |
| Eval | AIME'24, 30 problems × 8 samples (mean@8), 16 384 tokens, temperature 1.0, every 5 steps | same | AIME'24, 64 000 tokens, temperature 1.0, top-p 1.0, top-k off; temperature, top-p, top-k ✓, token cap ✗ |

All three differences in the tutorial column apply to both of our arms equally.
The exact resolved configs are in `runs/grpo/config.json` and
`runs/opd_matched/config.json`.

### Run logs (`runs/<arm>/`)

| File | Contents |
|---|---|
| `config.json` | resolved trainer config for steps 21–40 |
| `config_steps0to20.json` | resolved config for the first launch (differs only in `max_steps`, 21 vs. 41) |
| `metrics.jsonl` | per-step training and evaluation metrics, including AIME'24 score, pass@k, and truncation fraction |
| `logs.log` | trainer log for both stages, including the line `Loaded weights from tinker://b89f6676-…` at step 0 |
| `stdout_steps0to20.log`, `stdout_steps21to40.log` | console output of the two launches, including the exact command lines |
| `checkpoints.jsonl` | the tinker checkpoints the runs saved (steps 10, 20, 21, 30, 40, 41) |
| `code.diff` | diff of git-tracked files recorded at launch; identical for both arms and both stages |

For anonymization, local paths, the W&B host, the W&B username, and the W&B run
URLs in these files have been replaced with placeholders. Nothing else was
changed.

### Results

These numbers are from `runs/*/metrics.jsonl`, at step 40.

| | AIME'24 mean@8 | Eval rollouts truncated at 16 384 tokens |
|---|---|---|
| SFT init (step 0) | 0.500 | 49% |
| GRPO | 0.542 | 26.3% |
| OPD | 0.013 | 98.75% |

## Hyperparameter choices

The table above compares every setting with the official tutorial. In summary:

- **Taken from the tutorial or the script defaults:** the SFT checkpoint and
  its LoRA rank, the teacher, the training data (DeepMath-103K), the learning
  rate (1e-4, the tutorial's LoRA rate), the 16 384-token rollout cap, group
  size 4, temperature 1.0, the `importance_sampling` loss,
  `kl_penalty_coef=1.0`, and `kl_discount_factor=0`.
- **Our choice: batch size and step count.** The tutorial uses 512 prompts per
  batch for 200 steps (the script default is 1024). We used **64 prompts per
  batch for 40 steps.**
- **Our choice: prompt format.** The tutorial's OPD command uses
  `PromptOnlyDatasetBuilder`. It gives the bare DeepMath question, cuts the
  prompt at 1 024 tokens, and keeps the dataset order. Our GRPO arm needs a
  gradable answer, so it uses `MathEnv` prompts: the same question plus the
  instruction "Write your answer in \boxed{} format.", in shuffled order. We
  gave OPD the same prompts (`opd_dataset=matched`) so that the two arms see
  the same questions in the same order, and the training signal is the only
  difference. The launcher still supports the tutorial's format with
  `opd_dataset=prompt_only`, but we did not run it.
- **Our choice: evaluation cap.** See "Evaluation cap" below.

We chose the batch size and step count because we had limited compute credit.
We do not expect them to change the conclusion, for two reasons:

1. **Batch size.** The budget-controlled batch-size ablation in the paper
   (Appendix D.3, Figure 11) covers 64, 128, and 256 prompts at a fixed total
   of 12 800 prompts. Both methods gain roughly in parallel as the batch
   shrinks. GRPO stays above OPD at every batch size, and no OPD run overtakes
   GRPO.
2. **Step count.** By step 40, OPD has already collapsed. AIME'24 mean@8 falls
   from 0.500 to 0.013, and 98.75% of evaluation rollouts hit the length cap.
   From our experience with the verl runs in the main paper, an OPD run in this
   length-saturated state does not recover with further training.

GRPO and OPD use the same batch size and step count, so this choice does not
favor either one.

## Other details

- **LoRA rank.** Both `config.json` files show `"lora_rank": 32`. That is an
  unused default. When training starts from a checkpoint, tinker builds the
  training client from the checkpoint state (`create_training_client_from_state`)
  and ignores `lora_rank`. The saved adapters for both arms are rank 128, which
  is the rank of the SFT checkpoint.
- **Renderer.** We pinned the renderer to `renderer_name=qwen3_5`. The public
  SFT checkpoint has no renderer metadata, and its lookup returns 404.
  `qwen3_5` is the format that `off_policy_reasoning.py` uses to train that
  checkpoint, so the student, the teacher, and the SFT data all share one chat
  format.
- **Two-stage run.** The runs were launched with the default
  `num_train_steps=20`, then resumed to 40 with `num_train_steps=40
  behavior_if_log_dir_exists=resume`. The resume restores weights and
  optimizer state from the step-21 checkpoint. Batches are indexed by step, so
  the data order is unchanged. Apart from `max_steps` (21 → 41), the config is
  identical across the two stages.
- **Evaluation cap.** The tutorial reports AIME'24 with a 64 000-token cap. To
  save cost, we evaluate at 16 384 tokens, the same as the training cap. Both
  arms use the same evaluation settings.

## Changes to upstream

`patches/changes_vs_upstream_ab78b73.patch` contains every change. Running
`git apply` with it on a checkout of `ab78b73` reproduces `tinker-cookbook/`
exactly.

How this code relates to the code that ran:

- **The five modified upstream files** are byte-identical to the `code.diff`
  each run recorded. That diff was the same at the first launch and at the
  resume, for both arms.
- **The new files.** The recorded diff covers only files tracked by git, so it
  does not include the new files. The copies here were last modified between
  the first launch and the resume, so they are the code that ran steps 21–40.
  For steps 0–20, the resolved trainer config they produced is identical to
  the first launch's config, apart from `max_steps`.

New files:

- `recipes/distillation/opd_vs_grpo.py` launches both arms. It sets every
  shared hyperparameter in one place and checks that the renderer resolves to
  `qwen3_5`.
- `recipes/distillation/deepmath_zero_reward.py` defines
  `ZeroRewardMathEnv` / `ZeroRewardDeepMathDataset`. These give OPD the same
  DeepMath prompts, order, and grading as GRPO, but with zero reward, so the
  teacher KL is the only training signal. Grading still runs, so correctness is
  logged for both arms.
- `recipes/distillation/saving_benchmark_evaluator.py` is an evaluator that
  runs the AIME'24 benchmark at a fixed cadence and writes the generations to
  disk.
- Unit tests for the three files above, and for `distillation/datasets.py`.

Modified files:

- `eval/benchmarks/aime.py` registers `aime_2024`, from
  `HuggingFaceH4/aime_2024`.
- `distillation/train_on_policy.py` adds rollout logging, using the same
  `num_groups_to_log` / `rollout_json_export` options as the RL trainer. It also
  keeps builders, groups, and dataset indices aligned if a rollout group fails.
  No group failed in our runs.
- `distillation/datasets.py` makes `PromptOnlyEnv` log its transcripts. This
  only matters with `opd_dataset=prompt_only`, which we did not use.
- `rl/train.py` and `rl/rollout_logging.py` move the logtree helper so both
  trainers can share it. Behavior is unchanged.

None of these changes touch the GRPO advantage, the OPD reverse-KL advantage,
the loss function, or the optimizer.

## Reproducing

```bash
cd tinker-cookbook
uv sync --extra math-rl --extra wandb --extra dev
export TINKER_API_KEY=...        # a tinker account with Qwen3.5-9B access
export WANDB_API_KEY=...         # or append wandb_project=None to each command
cd .. && ./run.sh                # two-stage, as we ran it
# RUN_MODE=single ./run.sh       # equivalent single 40-step run
```

Tests: `cd tinker-cookbook && uv run pytest
tinker_cookbook/recipes/distillation/deepmath_zero_reward_test.py
tinker_cookbook/recipes/distillation/saving_benchmark_evaluator_test.py
tinker_cookbook/distillation/datasets_test.py` (24 tests).
