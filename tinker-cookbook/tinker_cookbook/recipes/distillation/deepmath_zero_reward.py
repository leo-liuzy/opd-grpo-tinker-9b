"""Zero-reward DeepMath dataset, for comparing on-policy distillation against RL.

On-policy distillation must take its entire training signal from the teacher KL:
:func:`tinker_cookbook.distillation.train_on_policy.incorporate_kl_penalty` *adds*
the KL advantage to whatever advantage the environment produced, so a nonzero
environment reward would silently turn distillation into "RL plus KL".

The stock distillation dataset (:class:`PromptOnlyDataset`) achieves a zero reward
by dropping the answer entirely, but it also asks a different question than
:class:`MathEnv` does: no ``\\boxed{}`` instruction, a 1024-token prompt
truncation, and no shuffle of the underlying rows. Those differences make it
unsuitable as a control for a DeepMath RL run.

This module instead derives its environments from :class:`DeepMathDataset`, so
the prompt text, the row filtering, and the shuffle order are identical to the
RL arm by construction. Only the reward is replaced. Grading still runs, so
DeepMath correctness is reported as a metric for both arms — the distillation
arm is measured on correctness without ever training on it.
"""

import dataclasses
from collections.abc import Sequence
from functools import partial

import chz

from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_env import DeepMathDataset, MathEnv
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


class ZeroRewardMathEnv(MathEnv):
    """A :class:`MathEnv` that grades the response but reports zero reward."""

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        result = await super().step(action, extra=extra)
        assert "correct" in result.metrics and "format" in result.metrics, (
            "MathEnv is expected to report 'correct' and 'format' metrics; "
            f"got {sorted(result.metrics)}. Grading must stay observable even "
            "though it no longer contributes to the reward."
        )
        return dataclasses.replace(result, reward=0.0)


class ZeroRewardDeepMathDataset(DeepMathDataset):
    """DeepMath with :class:`ZeroRewardMathEnv` in place of :class:`MathEnv`."""

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        builder = super()._make_env_group_builder(x, group_size)
        if builder is None:
            # Mirror the parent's row filtering rather than raising, so both arms
            # of a comparison train on exactly the same set of problems.
            return None
        env_thunk = builder.env_thunk
        assert isinstance(env_thunk, partial) and env_thunk.func is MathEnv, (
            "Expected DeepMathDataset to build MathEnv via functools.partial, got "
            f"{env_thunk!r}. Swapping in the zero-reward env relies on that shape; "
            "if the parent changed, this override must be updated to match."
        )
        return dataclasses.replace(
            builder,
            env_thunk=partial(ZeroRewardMathEnv, *env_thunk.args, **env_thunk.keywords),
        )

    def get_batch(self, index: int) -> Sequence[ProblemGroupBuilder]:
        builders = super().get_batch(index)
        assert len(builders) > 0, f"DeepMath batch {index} produced no environment groups"
        return builders  # pyright: ignore[reportReturnType]


@chz.chz
class ZeroRewardDeepMathDatasetBuilder(RLDatasetBuilder):
    """Mirrors ``DeepMathDatasetBuilder`` so both arms accept the same arguments."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[ZeroRewardDeepMathDataset, None]:
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"
        assert self.group_size > 0, f"group_size must be positive, got {self.group_size}"
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return ZeroRewardDeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            seed=self.seed,
        ), None
