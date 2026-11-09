"""Unit tests for the zero-reward DeepMath environment.

These tests use a scripted renderer and bypass the DeepMath download, so they
run without network access or a Tinker API key.
"""

import asyncio
from functools import partial
from typing import Any

import pytest
import tinker

from tinker_cookbook.recipes.distillation.deepmath_zero_reward import (
    ZeroRewardDeepMathDataset,
    ZeroRewardMathEnv,
)
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from tinker_cookbook.renderers import Message, ParseTermination

PROBLEM = "What is 2 + 2?"
ANSWER = "4"


class ScriptedRenderer:
    """Fake renderer: records the prompts it is asked to build, replays parses."""

    def __init__(self, parse_result: tuple[Message, ParseTermination]):
        self._parse_result = parse_result
        self.prompts: list[list[Message]] = []

    def get_stop_sequences(self) -> list[str]:
        return ["<stop>"]

    def build_generation_prompt(self, messages: list[Message], **kwargs: Any) -> tinker.ModelInput:
        self.prompts.append(list(messages))
        return tinker.ModelInput.from_ints([1, 2, 3])

    def parse_response(self, action: list[int]) -> tuple[Message, ParseTermination]:
        return self._parse_result


def _correct_response() -> tuple[Message, ParseTermination]:
    message: Message = {"role": "assistant", "content": f"The answer is \\boxed{{{ANSWER}}}"}
    return message, ParseTermination.STOP_SEQUENCE


def _make_envs() -> tuple[MathEnv, ZeroRewardMathEnv, ScriptedRenderer, ScriptedRenderer]:
    graded_renderer = ScriptedRenderer(_correct_response())
    zero_renderer = ScriptedRenderer(_correct_response())
    graded = MathEnv(PROBLEM, ANSWER, graded_renderer)  # pyright: ignore[reportArgumentType]
    zero = ZeroRewardMathEnv(PROBLEM, ANSWER, zero_renderer)  # pyright: ignore[reportArgumentType]
    return graded, zero, graded_renderer, zero_renderer


class TestPromptParity:
    def test_question_text_is_identical(self):
        graded, zero, _, _ = _make_envs()
        assert zero.get_question() == graded.get_question()
        assert "\\boxed{}" in zero.get_question()

    def test_generation_prompt_is_identical(self):
        graded, zero, graded_renderer, zero_renderer = _make_envs()
        asyncio.run(graded.initial_observation())
        asyncio.run(zero.initial_observation())
        assert zero_renderer.prompts == graded_renderer.prompts


class TestZeroReward:
    def test_graded_env_rewards_a_correct_answer(self):
        graded, _, _, _ = _make_envs()
        result = asyncio.run(graded.step([1, 2, 3]))
        assert result.reward > 0.0
        assert result.metrics["correct"] == 1.0

    def test_zero_reward_env_reports_no_reward(self):
        _, zero, _, _ = _make_envs()
        result = asyncio.run(zero.step([1, 2, 3]))
        assert result.reward == 0.0

    def test_grading_metrics_survive(self):
        graded, zero, _, _ = _make_envs()
        graded_result = asyncio.run(graded.step([1, 2, 3]))
        zero_result = asyncio.run(zero.step([1, 2, 3]))
        assert zero_result.metrics == graded_result.metrics
        assert zero_result.episode_done == graded_result.episode_done


class TestDatasetEnvSwap:
    """Exercises the env swap without loading the 103K-row DeepMath dataset."""

    def _dataset(self) -> ZeroRewardDeepMathDataset:
        dataset = object.__new__(ZeroRewardDeepMathDataset)
        dataset.renderer = ScriptedRenderer(_correct_response())  # pyright: ignore[reportAttributeAccessIssue]
        dataset.convo_prefix = None
        return dataset

    def test_swaps_in_zero_reward_env(self):
        builder = self._dataset()._make_env_group_builder(
            {"question": PROBLEM, "final_answer": ANSWER}, group_size=4
        )
        assert builder is not None
        assert builder.num_envs == 4
        assert builder.dataset_name == "deepmath"
        assert isinstance(builder.env_thunk, partial)
        assert builder.env_thunk.func is ZeroRewardMathEnv
        env = builder.env_thunk()
        assert isinstance(env, ZeroRewardMathEnv)
        assert env.get_question() == MathEnv(PROBLEM, ANSWER, env.renderer).get_question()

    @pytest.mark.parametrize(
        "row",
        [
            {"question": "", "final_answer": ANSWER},
            {"question": PROBLEM, "final_answer": ""},
        ],
    )
    def test_mirrors_parent_row_filtering(self, row: dict[str, str]):
        # Rows the RL arm drops must also be dropped here, so both arms of a
        # comparison train on the same problems.
        assert self._dataset()._make_env_group_builder(row, group_size=4) is None
