"""Tests for the per-step bookkeeping in :class:`SavingBenchmarkEvaluator`.

The evaluator infers the training step from its own call count, because the
trainers hand it nothing but a sampling client. These tests pin that inference
and the guard that fires when it stops holding.
"""

from pathlib import Path
from typing import cast

import pytest
import tinker

from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator
from tinker_cookbook.eval.benchmarks import BenchmarkResult
from tinker_cookbook.recipes.distillation import saving_benchmark_evaluator
from tinker_cookbook.recipes.distillation.saving_benchmark_evaluator import (
    SavingBenchmarkEvaluator,
)
from tinker_cookbook.renderers import Renderer

BENCHMARK = "aime_2024"


def _make_result(*, num_truncated: int = 0, num_examples: int = 240) -> BenchmarkResult:
    return BenchmarkResult(
        name=BENCHMARK,
        score=0.25,
        num_examples=num_examples,
        num_correct=60,
        num_errors=0,
        num_truncated=num_truncated,
        pass_at_k={1: 0.25, 8: 0.6},
    )


@pytest.fixture
def evaluator(tmp_path: Path) -> SavingBenchmarkEvaluator:
    return SavingBenchmarkEvaluator(
        BENCHMARK,
        cast(Renderer, object()),
        save_root=tmp_path,
        eval_every=5,
        num_samples=8,
        max_examples=200,
        max_tokens=16384,
        temperature=1.0,
    )


@pytest.fixture
def stub_benchmark(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Record the save_dir each run would have used, skipping the real benchmark."""
    seen_save_dirs: list[str | None] = []

    async def fake_call(self: BenchmarkEvaluator, _: tinker.SamplingClient) -> dict[str, float]:
        seen_save_dirs.append(self.config.save_dir)
        assert self.config.save_dir is not None
        # The real runner creates this directory; later calls assert on its absence.
        (Path(self.config.save_dir) / self.benchmark_name).mkdir(parents=True)
        return {f"eval/{self.benchmark_name}/score": 0.25}

    monkeypatch.setattr(BenchmarkEvaluator, "__call__", fake_call)
    monkeypatch.setattr(
        saving_benchmark_evaluator, "load_result", lambda *_args, **_kwargs: _make_result()
    )
    return seen_save_dirs


def test_num_samples_reaches_the_benchmark_config(evaluator: SavingBenchmarkEvaluator) -> None:
    assert evaluator.config.num_samples == 8


def test_timeout_stays_above_the_runner_default(evaluator: SavingBenchmarkEvaluator) -> None:
    # A 16k-token rollout can outlast BenchmarkConfig's 300s default, and a
    # timed-out example is scored wrong rather than truncated.
    assert evaluator.config.timeout_seconds >= 1800


@pytest.mark.parametrize(("num_samples", "eval_every"), [(0, 5), (1, 0), (-1, 5)])
def test_rejects_degenerate_settings(tmp_path: Path, num_samples: int, eval_every: int) -> None:
    with pytest.raises(AssertionError):
        SavingBenchmarkEvaluator(
            BENCHMARK,
            cast(Renderer, object()),
            save_root=tmp_path,
            eval_every=eval_every,
            num_samples=num_samples,
        )


@pytest.mark.asyncio
async def test_save_dir_tracks_the_training_step(
    evaluator: SavingBenchmarkEvaluator, stub_benchmark: list[str | None], tmp_path: Path
) -> None:
    sampling_client = cast(tinker.SamplingClient, object())
    for _ in range(3):
        await evaluator(sampling_client)

    assert stub_benchmark == [
        str(tmp_path / "step_0000"),
        str(tmp_path / "step_0005"),
        str(tmp_path / "step_0010"),
    ]


@pytest.mark.asyncio
async def test_resumed_run_labels_steps_from_first_eval_step(
    stub_benchmark: list[str | None], tmp_path: Path
) -> None:
    # A run resumed at step 21 does not evaluate at 21: the trainers only eval
    # where i_batch % eval_every == 0, so the next one lands at 25.
    resumed = SavingBenchmarkEvaluator(
        BENCHMARK,
        cast(Renderer, object()),
        save_root=tmp_path,
        eval_every=5,
        first_eval_step=25,
        num_samples=8,
    )
    for _ in range(2):
        await resumed(cast(tinker.SamplingClient, object()))

    assert stub_benchmark == [str(tmp_path / "step_0025"), str(tmp_path / "step_0030")]


def test_first_eval_step_must_align_with_eval_every(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="multiple of"):
        SavingBenchmarkEvaluator(
            BENCHMARK,
            cast(Renderer, object()),
            save_root=tmp_path,
            eval_every=5,
            first_eval_step=21,
        )


@pytest.mark.asyncio
async def test_refuses_to_reuse_a_step_directory(
    evaluator: SavingBenchmarkEvaluator, stub_benchmark: list[str | None], tmp_path: Path
) -> None:
    # Stand in for a resumed run, whose call count restarts at 0 while the real
    # training step does not. Pooling two checkpoints' generations into one
    # trajectories.jsonl must fail loudly instead.
    (tmp_path / "step_0000" / BENCHMARK).mkdir(parents=True)

    with pytest.raises(AssertionError, match="no longer identifies the training step"):
        await evaluator(cast(tinker.SamplingClient, object()))
    assert stub_benchmark == []


@pytest.mark.asyncio
async def test_reports_truncation_and_pass_at_k(
    evaluator: SavingBenchmarkEvaluator,
    stub_benchmark: list[str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        saving_benchmark_evaluator,
        "load_result",
        lambda *_args, **_kwargs: _make_result(num_truncated=24, num_examples=240),
    )

    metrics = await evaluator(cast(tinker.SamplingClient, object()))

    prefix = f"eval/{BENCHMARK}"
    assert metrics[f"{prefix}/num_truncated"] == 24.0
    assert metrics[f"{prefix}/frac_truncated"] == pytest.approx(0.1)
    assert metrics[f"{prefix}/pass@1"] == pytest.approx(0.25)
    assert metrics[f"{prefix}/pass@8"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_missing_result_file_is_an_error(
    evaluator: SavingBenchmarkEvaluator,
    stub_benchmark: list[str | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(saving_benchmark_evaluator, "load_result", lambda *_args, **_kwargs: None)

    with pytest.raises(AssertionError, match=r"wrote no result\.json"):
        await evaluator(cast(tinker.SamplingClient, object()))
