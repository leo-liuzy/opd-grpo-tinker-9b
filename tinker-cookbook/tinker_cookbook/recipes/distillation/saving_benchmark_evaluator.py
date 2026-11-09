"""Inline benchmark evaluator that keeps its generations and reports truncation.

:class:`~tinker_cookbook.eval.benchmark_evaluator.BenchmarkEvaluator` reports a
score and discards the generations, which leaves an eval result hard to audit:
a wrong answer and an answer cut off at ``max_tokens`` are indistinguishable in
the metrics. This subclass adds three things on top of it:

* saves decoded transcripts and ``result.json`` under one directory per eval
  step, so a finished run still contains what the model actually wrote;
* draws ``num_samples`` per example, which turns the reported score into a mean
  over that many draws and shrinks run-to-run noise on small benchmarks;
* surfaces truncation counts and pass@k, which :class:`BenchmarkResult` carries
  as top-level fields that the base evaluator drops.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import tinker

from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator
from tinker_cookbook.eval.benchmarks import load_result
from tinker_cookbook.renderers import Renderer

logger = logging.getLogger(__name__)


class SavingBenchmarkEvaluator(BenchmarkEvaluator):
    """A :class:`BenchmarkEvaluator` that persists generations under a per-step directory.

    Args:
        benchmark_name: Name of the benchmark in the registry (e.g. ``"aime_2024"``).
        renderer: Renderer for tokenization and prompt building.
        save_root: Directory to hold one ``step_XXXX/`` subdirectory per eval.
        eval_every: The training loop's eval interval, used to label the
            per-step directories.
        first_eval_step: Training step of this evaluator's first call. Zero for
            a fresh run; on a resumed run it must be the first multiple of
            ``eval_every`` at or after the resumed start step, since that is
            when the trainers' ``i_batch % eval_every == 0`` check next fires.
        num_samples: Draws per example. Values above 1 make the reported
            ``score`` a mean over that many draws (``pass@1`` in the pass@k
            table is the same quantity), which is the cheap way to get a usable
            signal out of a benchmark with few examples.
        max_examples: Maximum examples to evaluate per call.
        max_tokens: Maximum generation tokens per sample.
        temperature: Sampling temperature.
        timeout_seconds: Per-example wall-clock limit. Examples that exceed it
            are scored as failures rather than truncations, so this must stay
            comfortably above the time it takes to generate ``max_tokens``.
    """

    def __init__(
        self,
        benchmark_name: str,
        renderer: Renderer,
        *,
        save_root: Path | str,
        eval_every: int,
        first_eval_step: int = 0,
        num_samples: int = 1,
        max_examples: int = 100,
        max_tokens: int = 32768,
        temperature: float = 0.6,
        timeout_seconds: float = 1800,
    ):
        super().__init__(
            benchmark_name,
            renderer,
            max_examples=max_examples,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assert num_samples >= 1, f"num_samples must be >= 1, got {num_samples}"
        assert eval_every >= 1, f"eval_every must be >= 1, got {eval_every}"
        assert first_eval_step >= 0 and first_eval_step % eval_every == 0, (
            f"first_eval_step ({first_eval_step}) must be a non-negative multiple of "
            f"eval_every ({eval_every}); the trainers only evaluate on steps where "
            "i_batch % eval_every == 0"
        )
        self.config = dataclasses.replace(
            self.config, num_samples=num_samples, timeout_seconds=timeout_seconds
        )
        self._save_root = Path(save_root)
        self._eval_every = eval_every
        self._first_eval_step = first_eval_step
        self._num_evals = 0

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Run the benchmark, save its transcripts, and return metrics.

        Returns the base evaluator's metrics plus ``num_truncated``,
        ``frac_truncated``, and one ``pass@k`` entry per reported k.
        """
        # Both trainers build the evaluator once and call it at each eval step in
        # order, so the call count plus first_eval_step recovers the step number.
        # The directory assertion below is what keeps that inference honest.
        step = self._first_eval_step + self._num_evals * self._eval_every
        self._num_evals += 1
        save_dir = self._save_root / f"step_{step:04d}"
        benchmark_dir = save_dir / self.benchmark_name
        assert not benchmark_dir.exists(), (
            f"{benchmark_dir} already exists, so this evaluator's call count "
            f"({self._num_evals}) plus first_eval_step ({self._first_eval_step}) no "
            "longer identifies the training step. Benchmark runs append to "
            "trajectories.jsonl, which would pool generations from two different "
            "checkpoints into one file."
        )
        self.config = dataclasses.replace(self.config, save_dir=str(save_dir))

        metrics = await super().__call__(sampling_client)

        result = load_result(str(save_dir), self.benchmark_name)
        assert result is not None, (
            f"{self.benchmark_name} reported metrics but wrote no result.json under "
            f"{save_dir}; the transcripts this run is supposed to keep are missing"
        )
        assert result.num_examples > 0, (
            f"{self.benchmark_name} graded 0 examples at step {step}; "
            f"max_examples={self.config.max_examples}"
        )
        prefix = f"eval/{self.benchmark_name}"
        metrics[f"{prefix}/num_truncated"] = float(result.num_truncated)
        metrics[f"{prefix}/frac_truncated"] = result.num_truncated / result.num_examples
        for k, value in result.pass_at_k.items():
            metrics[f"{prefix}/pass@{k}"] = float(value)

        logger.info(
            "%s step %d: score=%.3f over %d graded samples, %d truncated at %d tokens; "
            "transcripts in %s",
            self.benchmark_name,
            step,
            result.score,
            result.num_examples,
            result.num_truncated,
            self.config.max_tokens,
            benchmark_dir,
        )
        return metrics
