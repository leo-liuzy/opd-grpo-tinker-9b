"""Tests for the distillation environments' zero reward and rollout logging."""

import json
from pathlib import Path

import pytest

from tinker_cookbook import renderers
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.renderers import ParseTermination
from tinker_cookbook.utils import logtree

PROMPT = "What is the smallest prime factor of 2027?"
RESPONSE = "The number 2027 is prime, so the answer is 2027."


class StubRenderer:
    """Minimal renderer: parses a fixed response, avoiding a tokenizer download."""

    def __init__(self, termination: ParseTermination = ParseTermination.STOP_SEQUENCE):
        self._termination = termination

    def parse_response(self, response: list[int]) -> tuple[renderers.Message, ParseTermination]:
        assert response, "expected a non-empty action"
        return {"role": "assistant", "content": RESPONSE}, self._termination

    def get_stop_sequences(self) -> list[int]:
        return []


def _make_env(termination: ParseTermination = ParseTermination.STOP_SEQUENCE) -> PromptOnlyEnv:
    renderer = StubRenderer(termination)
    return PromptOnlyEnv(PROMPT, renderer)  # pyright: ignore[reportArgumentType]


async def _step_and_capture(env: PromptOnlyEnv, tmp_path: Path) -> tuple[float, str]:
    """Run one step inside a logtree trace, returning the reward and the trace text."""
    html_path = tmp_path / "train.html"
    with logtree.init_trace("test", path=str(html_path)) as trace:
        result = await env.step([1, 2, 3])
    return result.reward, json.dumps(trace.to_dict())


@pytest.mark.asyncio
async def test_reward_is_zero(tmp_path: Path) -> None:
    # The KL penalty is added to whatever the env returns, so any nonzero reward
    # here would quietly turn distillation into "GRPO + KL".
    reward, _ = await _step_and_capture(_make_env(), tmp_path)
    assert reward == 0.0


@pytest.mark.asyncio
async def test_reports_no_grading_metrics(tmp_path: Path) -> None:
    # check_answer() is hardwired to False, so emitting "correct" would read as
    # "never correct" rather than "never graded".
    env = _make_env()
    with logtree.init_trace("test", path=str(tmp_path / "train.html")):
        result = await env.step([1, 2, 3])
    assert result.metrics == {}


@pytest.mark.asyncio
async def test_logs_prompt_and_response_text(tmp_path: Path) -> None:
    # Distillation supervises on tokens and grades nothing, so this trace is the
    # only place a finished run records what the student actually wrote.
    _, trace_text = await _step_and_capture(_make_env(), tmp_path)
    assert PROMPT in trace_text
    assert RESPONSE in trace_text


@pytest.mark.asyncio
async def test_logs_termination_so_truncation_is_visible(tmp_path: Path) -> None:
    _, clean_text = await _step_and_capture(_make_env(), tmp_path)
    assert ParseTermination.STOP_SEQUENCE.value in clean_text

    # A rollout cut off at max_tokens never emits a stop sequence.
    _, truncated_text = await _step_and_capture(
        _make_env(ParseTermination.MALFORMED), tmp_path / "truncated"
    )
    assert ParseTermination.MALFORMED.value in truncated_text


@pytest.mark.asyncio
async def test_writes_html_transcript(tmp_path: Path) -> None:
    env = _make_env()
    html_path = tmp_path / "train.html"
    with logtree.init_trace("test", path=str(html_path)):
        await env.step([1, 2, 3])
    assert html_path.exists(), "logtree should have written the HTML transcript"
    assert RESPONSE in html_path.read_text()
