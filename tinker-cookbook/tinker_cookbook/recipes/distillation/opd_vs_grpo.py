"""Compare on-policy distillation against GRPO on DeepMath from a shared SFT init.

Both arms are launched from this one entrypoint so that every knob they have in
common — student checkpoint, group size, rollout length, learning rate,
temperature, evaluation — is set in exactly one place and cannot drift between
runs. The intended difference is the training signal:

  ``method=grpo``  correctness/format reward from ``MathEnv``, no teacher.
  ``method=opd``   zero reward, advantages come entirely from the teacher KL.

``opd_dataset`` controls a real tradeoff between controlling the comparison and
reproducing this recipe's README:

  ``matched`` (default) drives both arms from the same ``DeepMathDataset``, so
      the question text (including ``MathEnv``'s ``\\boxed{}`` instruction),
      problem order, prompt truncation, and grading are all identical, leaving
      the training signal as the only difference. The distillation arm runs
      ``ZeroRewardMathEnv``, which grades for metrics but reports zero reward.
      Deviates from the README's dataset handling.

  ``prompt_only`` is what the README's on-policy distillation command runs: bare
      DeepMath questions via ``PromptOnlyDatasetBuilder``, truncated to 1024
      prompt tokens, in raw dataset order. Faithful to the published recipe, but
      the GRPO arm then asks a different question over a different, shuffled
      slice of DeepMath, so prompt text and problem selection are confounded
      with the training signal, and the OPD arm reports no correctness metric.

Example usage:
    export TINKER_API_KEY=...   # billed account; also holds the checkpoints
    export WANDB_API_KEY=...    # required unless wandb_project=None

    python -m tinker_cookbook.recipes.distillation.opd_vs_grpo method=grpo
    python -m tinker_cookbook.recipes.distillation.opd_vs_grpo method=opd

Note that the LoRA rank is inherited from ``load_checkpoint_path`` — both
trainers call ``create_training_client_from_state_async``, which ignores the
``lora_rank`` field. The default checkpoint below is the rank-128 SFT run
published in this recipe's README.
"""

import asyncio
import importlib.util
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)
from tinker_cookbook.eval.benchmarks import REGISTRY
from tinker_cookbook.eval.benchmarks._runner import _ensure_registered
from tinker_cookbook.eval.evaluators import SamplingClientEvaluatorBuilder
from tinker_cookbook.recipes.distillation.deepmath_zero_reward import (
    ZeroRewardDeepMathDatasetBuilder,
)
from tinker_cookbook.recipes.distillation.saving_benchmark_evaluator import (
    SavingBenchmarkEvaluator,
)
from tinker_cookbook.recipes.math_rl.math_env import DeepMathDatasetBuilder
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# Rank-128 supervised fine-tuning checkpoint from recipes/distillation/README.md.
SFT_RANK128_CHECKPOINT = "tinker://b89f6676-49c4-53ed-9788-2e0c81a9eabf:train:0/weights/final"

# Deliberately not under /tmp: these runs cost real money and their logs, rollout
# transcripts, and eval generations are the only local record of them, while
# macOS prunes /tmp periodically and can clear it on reboot.
LOG_ROOT = "~/tinker-runs/opd_vs_grpo"

Method = Literal["opd", "grpo"]
OPDDataset = Literal["prompt_only", "matched"]


@chz.chz
class CLIConfig:
    """Shared configuration for both arms of the comparison."""

    method: Method

    # Models. The student starts from an SFT checkpoint; the teacher is only
    # used by the distillation arm.
    student_model: str = "Qwen/Qwen3.5-9B-Base"
    teacher_model: str = "Qwen/Qwen3.5-9B"
    load_checkpoint_path: str = SFT_RANK128_CHECKPOINT
    teacher_checkpoint: str | None = None
    # Left as None, the renderer is read from the checkpoint's metadata.
    renderer_name: str | None = None
    # The renderer the SFT checkpoint was trained with. off_policy_reasoning.py
    # pins "qwen3_5" rather than the base model's recommended renderer
    # ("role_colon"), so that the student shares a format with the instruct
    # teacher — the teacher's logprobs are computed on the student's own
    # rendered tokens. Resolving to anything else means the checkpoint metadata
    # is missing and we silently fell back to the base default, which would
    # make the KL meaningless. Set to None to skip the check.
    expected_renderer_name: str | None = "qwen3_5"

    # Which dataset the distillation arm uses. See the module docstring for the
    # comparison-control vs README-fidelity tradeoff.
    opd_dataset: OPDDataset = "matched"

    # Training. Identical for both arms. The README ties its reported score to
    # 16k-token rollouts.
    groups_per_batch: int = 64
    group_size: int = 4
    max_tokens: int = 16384
    learning_rate: float = 1e-4
    temperature: float = 1.0
    num_train_steps: int = 20
    seed: int = 0

    # Distillation-only. The KL is the entire training signal for method=opd.
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Evaluation. Runs at step 0 (the SFT baseline) and every eval_every steps.
    # aime_2024 is the metric the README reports. It has only 30 problems, so a
    # single draw per problem is far too noisy to resolve 20 steps of training:
    # one problem moves the score by 3.3%, and the run-to-run spread at
    # temperature 1.0 is wider than any effect we expect. eval_num_samples
    # averages several draws per problem instead (the reported score becomes the
    # mean over draws), which cuts that noise for a few dollars.
    eval_benchmarks: tuple[str, ...] = ("aime_2024",)
    eval_every: int = 5
    eval_max_examples: int = 200
    eval_num_samples: int = 8
    # Kept at the training rollout length so eval is not the tighter cap. The
    # README evaluated AIME'24 at 64000 tokens; raise this to close that gap.
    eval_max_tokens: int = 16384
    # The README's other three eval settings are matched: temperature=1.0 here,
    # and top_p=1.0 / top_k=-1 are the SDK defaults that TinkerTokenCompleter
    # leaves untouched.
    eval_temperature: float = 1.0
    # BenchmarkConfig defaults to 300s, which a 16k-token reasoning rollout can
    # exceed. A timed-out example is scored 0 rather than counted as truncated,
    # and it would hit whichever arm generates longer outputs harder, so the
    # limit has to sit well above the time eval_max_tokens actually takes.
    eval_timeout_seconds: float = 1800

    save_every: int = 10
    log_path: str | None = None
    # On by default: the local log directory is the only other record of a run
    # that costs real money, and W&B is account-independent of the Tinker key —
    # it streams to whoever owns WANDB_API_KEY. Scalars and the resolved config
    # only; rollout text stays in local files. Set to None to disable.
    wandb_project: str | None = "opd_vs_grpo"
    wandb_name: str | None = None
    # Trajectory groups whose prompt and response *text* is written to
    # ``iteration_*/train.html`` and its logtree JSON. The rollout summaries
    # (rewards, lengths) always cover the full batch; this is the text. None
    # means every group, which is what makes the generations reviewable rather
    # than just countable — at 16k-token rollouts that is a few hundred MB per
    # arm. Set an integer to keep only the first N groups per step.
    num_groups_to_log: int | None = None
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _build_opd_dataset_builder(cli_config: CLIConfig, renderer_name: str) -> RLDatasetBuilder:
    """Build the distillation arm's dataset, honoring the README/control tradeoff."""
    assert cli_config.opd_dataset in ("prompt_only", "matched"), (
        f"unknown opd_dataset {cli_config.opd_dataset!r}"
    )
    if cli_config.opd_dataset == "prompt_only":
        # The README's command: bare prompts, 1024-token truncation, dataset order.
        return PromptOnlyDatasetBuilder(
            dataset_name="deepmath",
            groups_per_batch=cli_config.groups_per_batch,
            group_size=cli_config.group_size,
            model_name_for_tokenizer=cli_config.student_model,
            renderer_name=renderer_name,
        )
    return ZeroRewardDeepMathDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.student_model,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
    )


def _first_eval_step(log_path: str, eval_every: int) -> int:
    """Return the training step at which the next run's first evaluation lands.

    Both trainers resume by reading the last record in ``checkpoints.jsonl``
    under ``log_path`` and evaluating on steps where ``i_batch % eval_every == 0``.
    Resolving the same number here keeps the evaluator's per-step output
    directories labeled with real training steps across a resume, instead of
    restarting at 0 and colliding with the previous run's directories.
    """
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    start_batch = resume_info.batch if resume_info else 0
    assert start_batch >= 0, f"resumed start step must be non-negative, got {start_batch}"
    if start_batch > 0:
        logger.info(
            "Found a checkpoint at step %d in %s; evaluations will resume at step %d",
            start_batch,
            log_path,
            math.ceil(start_batch / eval_every) * eval_every,
        )
    return math.ceil(start_batch / eval_every) * eval_every


def _build_evaluator_builders(
    cli_config: CLIConfig, renderer: renderers.Renderer, log_path: str
) -> list[SamplingClientEvaluatorBuilder]:
    assert len(cli_config.eval_benchmarks) > 0, (
        "at least one eval benchmark is required; the two arms need a comparable metric"
    )
    # REGISTRY is populated lazily by module import, so resolve each name before
    # deciding it is unknown — this is what run_benchmark does internally.
    for name in cli_config.eval_benchmarks:
        _ensure_registered(name)
    unknown = [name for name in cli_config.eval_benchmarks if name not in REGISTRY]
    assert not unknown, (
        f"unknown benchmark(s) {unknown}; registered benchmarks are {sorted(REGISTRY)}"
    )

    first_eval_step = _first_eval_step(log_path, cli_config.eval_every)

    def make_builder(name: str) -> SamplingClientEvaluatorBuilder:
        return lambda: SavingBenchmarkEvaluator(
            name,
            renderer,
            save_root=Path(log_path) / "evals" / name,
            eval_every=cli_config.eval_every,
            first_eval_step=first_eval_step,
            num_samples=cli_config.eval_num_samples,
            max_examples=cli_config.eval_max_examples,
            max_tokens=cli_config.eval_max_tokens,
            temperature=cli_config.eval_temperature,
            timeout_seconds=cli_config.eval_timeout_seconds,
        )

    return [make_builder(name) for name in cli_config.eval_benchmarks]


async def build_config(
    cli_config: CLIConfig,
) -> rl_train.Config | train_on_policy.Config:
    """Resolve the CLI config into the trainer config for the requested arm."""
    assert cli_config.method in ("opd", "grpo"), f"unknown method {cli_config.method!r}"
    assert cli_config.num_train_steps > 0, "num_train_steps must be positive"
    assert cli_config.eval_every > 0, (
        "eval_every must be positive; without evaluation the two arms have no comparable metric"
    )
    # Both trainers evaluate at the top of iteration i_batch, before that
    # iteration trains. Running one extra iteration therefore yields an
    # evaluation that reflects exactly num_train_steps completed updates.
    assert cli_config.num_train_steps % cli_config.eval_every == 0, (
        f"eval_every ({cli_config.eval_every}) must divide num_train_steps "
        f"({cli_config.num_train_steps}) so the final evaluation lands on the last step"
    )
    max_steps = cli_config.num_train_steps + 1

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.student_model,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    if cli_config.expected_renderer_name is not None:
        assert renderer_name == cli_config.expected_renderer_name, (
            f"resolved renderer {renderer_name!r} but expected "
            f"{cli_config.expected_renderer_name!r} for checkpoint "
            f"{cli_config.load_checkpoint_path}. The student and teacher must share a "
            "render format for the teacher KL to be meaningful. Pass "
            "renderer_name=<name> to override, or expected_renderer_name=None to skip "
            "this check."
        )
    tokenizer = get_tokenizer(cli_config.student_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    if cli_config.log_path is not None:
        log_path = str(Path(cli_config.log_path).expanduser())
    else:
        student = cli_config.student_model.replace("/", "-")
        arm = f"opd_{cli_config.opd_dataset}" if cli_config.method == "opd" else cli_config.method
        run_name = (
            f"{arm}-deepmath-{student}-"
            f"{cli_config.learning_rate}lr-{cli_config.groups_per_batch}x{cli_config.group_size}-"
            f"{cli_config.max_tokens}tok-seed{cli_config.seed}-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = str(Path(LOG_ROOT).expanduser() / run_name)
    # Everything downstream joins onto this path, so it has to be expanded here
    # rather than relying on the trainer Config's munger.
    assert "~" not in log_path, f"log_path was not expanded: {log_path}"
    wandb_name = cli_config.wandb_name or Path(log_path).name
    evaluator_builders = _build_evaluator_builders(cli_config, renderer, log_path)

    # None means "every group"; resolved here so the value recorded in
    # config.json is the literal number of groups logged.
    num_groups_to_log = (
        cli_config.groups_per_batch
        if cli_config.num_groups_to_log is None
        else cli_config.num_groups_to_log
    )
    assert 0 <= num_groups_to_log <= cli_config.groups_per_batch, (
        f"num_groups_to_log ({num_groups_to_log}) must be between 0 and "
        f"groups_per_batch ({cli_config.groups_per_batch})"
    )

    logger.info(
        "Launching %s: student=%s from %s, %d groups x %d rollouts, max_tokens=%d, "
        "lr=%s, %d steps (LoRA rank is inherited from the checkpoint)",
        cli_config.method,
        cli_config.student_model,
        cli_config.load_checkpoint_path,
        cli_config.groups_per_batch,
        cli_config.group_size,
        cli_config.max_tokens,
        cli_config.learning_rate,
        cli_config.num_train_steps,
    )

    if cli_config.method == "grpo":
        config = rl_train.Config(
            learning_rate=cli_config.learning_rate,
            dataset_builder=DeepMathDatasetBuilder(
                batch_size=cli_config.groups_per_batch,
                model_name_for_tokenizer=cli_config.student_model,
                renderer_name=renderer_name,
                group_size=cli_config.group_size,
                seed=cli_config.seed,
            ),
            model_name=cli_config.student_model,
            recipe_name="recipe_distillation_opd_vs_grpo",
            renderer_name=renderer_name,
            max_tokens=cli_config.max_tokens,
            temperature=cli_config.temperature,
            load_checkpoint_path=cli_config.load_checkpoint_path,
            evaluator_builders=evaluator_builders,
            # Keep every group, so the effective batch size matches the
            # distillation arm (whose rewards are constant by construction).
            remove_constant_reward_groups=False,
            eval_every=cli_config.eval_every,
            save_every=cli_config.save_every,
            max_steps=max_steps,
            log_path=log_path,
            num_groups_to_log=num_groups_to_log,
            wandb_project=cli_config.wandb_project,
            wandb_name=wandb_name,
            base_url=cli_config.base_url,
        )
    else:
        assert cli_config.kl_penalty_coef > 0, (
            "on-policy distillation takes its entire training signal from the teacher KL, "
            f"but kl_penalty_coef={cli_config.kl_penalty_coef}"
        )
        config = train_on_policy.Config(
            learning_rate=cli_config.learning_rate,
            dataset_configs=[
                DistillationDatasetConfig(
                    dataset_builder=_build_opd_dataset_builder(cli_config, renderer_name),
                    teacher_config=TeacherConfig(
                        base_model=cli_config.teacher_model,
                        load_checkpoint_path=cli_config.teacher_checkpoint,
                    ),
                    groups_per_batch=cli_config.groups_per_batch,
                )
            ],
            model_name=cli_config.student_model,
            recipe_name="recipe_distillation_opd_vs_grpo",
            renderer_name=renderer_name,
            max_tokens=cli_config.max_tokens,
            temperature=cli_config.temperature,
            load_checkpoint_path=cli_config.load_checkpoint_path,
            evaluator_builders=evaluator_builders,
            kl_penalty_coef=cli_config.kl_penalty_coef,
            kl_discount_factor=cli_config.kl_discount_factor,
            eval_every=cli_config.eval_every,
            save_every=cli_config.save_every,
            max_steps=max_steps,
            log_path=log_path,
            num_groups_to_log=num_groups_to_log,
            wandb_project=cli_config.wandb_project,
            wandb_name=wandb_name,
            base_url=cli_config.base_url,
        )

    return config


def _assert_wandb_ready(wandb_project: str | None) -> None:
    """Fail before launch if W&B is requested but cannot actually log.

    ``ml_log.setup_logging`` degrades quietly here: a missing ``wandb`` package
    or an unset ``WANDB_API_KEY`` only prints a warning and drops the logger,
    which scrolls past in startup noise. The failure would then surface hours
    and many dollars later as an empty project, so this turns it into a startup
    error instead.
    """
    if wandb_project is None:
        return
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError(
            f"wandb_project={wandb_project!r} but the wandb package is not installed; "
            "run `uv sync --extra wandb` (or pass wandb_project=None to opt out)"
        )
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"wandb_project={wandb_project!r} but WANDB_API_KEY is not set, so metrics "
            "would silently go nowhere. Export the key from wandb.ai/authorize "
            "(or pass wandb_project=None to opt out)."
        )
    # A "local-" key belongs to a self-hosted W&B Server. Without WANDB_BASE_URL,
    # the client talks to https://api.wandb.ai, which rejects it with a 401 that
    # reads like a bad key rather than a misrouted one.
    if api_key.startswith("local-") and not os.environ.get("WANDB_BASE_URL"):
        raise RuntimeError(
            "WANDB_API_KEY is a self-hosted ('local-') key but WANDB_BASE_URL is not "
            "set, so the client would authenticate against https://api.wandb.ai and "
            "fail with 401. Export WANDB_BASE_URL=https://<your-wandb-server>."
        )


async def cli_main(cli_config: CLIConfig):
    _assert_wandb_ready(cli_config.wandb_project)
    config = await build_config(cli_config)
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    if isinstance(config, rl_train.Config):
        await rl_train.main(config)
    else:
        await train_on_policy.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
