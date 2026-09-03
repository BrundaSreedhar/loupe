"""Token accounting.

The point of recording these is that two of this project's standing questions —
whether four reviewers beat one, and whether warming the shared prefix does what
the architecture claims — are questions about spend. So the numbers have to be
right, and a run that recorded nothing must not read as a run that cost nothing.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from loupe.tokens import Meter, collect


def reply(model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0,
          reasoning: int = 0) -> LLMResult:
    """One model response, shaped the way LangChain hands it to a callback."""
    message = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
            "output_token_details": {"reasoning": reasoning},
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_counts_are_totalled_and_kept_per_model():
    meter = Meter()
    meter.on_llm_end(reply("reviewer-model", 1000, 200))
    meter.on_llm_end(reply("reviewer-model", 1500, 300))
    meter.on_llm_end(reply("verifier-model", 800, 100))

    tokens = collect(meter)

    assert tokens.calls == 3
    assert tokens.input == 3300
    assert tokens.output == 600
    assert tokens.total == 3900
    # Roles are pointed at different models on purpose, to get separate daily
    # allowances. One total would hide which allowance the spend came from.
    assert tokens.by_model["reviewer-model"]["input"] == 2500
    assert tokens.by_model["verifier-model"]["input"] == 800


def test_cache_reads_and_writes_are_counted_apart():
    """The warm is supposed to turn four writes of one prefix into one write and
    three reads. Summing them together would make that unmeasurable."""
    meter = Meter()
    meter.on_llm_end(reply("m", 10_000, 100, cache_write=9_000))
    for _ in range(3):
        meter.on_llm_end(reply("m", 10_000, 100, cache_read=9_000))

    tokens = collect(meter)

    assert tokens.cache_write == 9_000
    assert tokens.cache_read == 27_000
    assert tokens.cache_hit_rate == 0.75


def test_thinking_tokens_are_visible():
    meter = Meter()
    meter.on_llm_end(reply("m", 100, 900, reasoning=800))
    assert collect(meter).reasoning == 800


def test_nothing_recorded_is_not_the_same_as_nothing_spent():
    empty = collect(Meter())
    assert empty.calls == 0
    assert empty.total == 0
    # No cacheable input at all is a hit rate of zero, and must not be confused
    # with a cache that was offered work and missed.
    assert empty.cache_hit_rate == 0.0


def test_a_review_carries_what_it_cost(monkeypatch):
    """The wiring: the meter goes into the graph config, and what it saw comes
    back on the result."""
    import loupe.runner as runner
    from loupe.schema import ReviewRequest

    def fake_invoke(request, mode, verify, run_name, remember, lint, meter, progress=None):
        meter.on_llm_end(reply("some-model", 4000, 500, cache_read=3000))
        return {"findings": [], "merged": [], "accepted": [], "verdicts": []}

    monkeypatch.setattr(runner, "_invoke", fake_invoke)
    result = runner.run_review(ReviewRequest(source="local", ref="HEAD"))

    assert result.tokens.calls == 1
    assert result.tokens.input == 4000
    assert result.tokens.cache_read == 3000


def test_an_eval_report_says_when_usage_was_never_reported():
    """A provider that reports no usage would otherwise show as an arm that ran
    for free, which is the same class of lie as reporting an unreviewed case as
    clean."""
    from evals.scoring import CaseScore, Report

    silent = Report(scores=[CaseScore("a", "defect", True, 0, 1, 1, 1, 0.0)])
    assert silent.metered == 0
    assert silent.tokens_per_review == 0.0

    measured = Report(scores=[
        CaseScore("a", "defect", True, 0, 1, 1, 1, 0.0, calls=3, tokens_in=900, tokens_out=100)
    ])
    assert measured.metered == 1
    assert measured.tokens_per_review == 1000


def test_anthropic_cache_writes_are_not_reported_as_zero():
    """Checked against the installed package rather than assumed.

    When Anthropic returns the per-TTL breakdown, langchain_anthropic moves the
    numbers into `ephemeral_5m_input_tokens` and sets `cache_creation` to 0 so the
    two are not double-counted. Reading only `cache_creation` therefore reports
    every Anthropic cache write as zero — which reads as "warming the prefix is
    free", the one conclusion this measurement exists to test.
    """
    message = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 9000,
            "output_tokens": 20,
            "total_tokens": 9020,
            "input_token_details": {
                "cache_read": 0,
                "cache_creation": 0,
                "ephemeral_5m_input_tokens": 9000,
                "ephemeral_1h_input_tokens": 0,
            },
        },
        response_metadata={"model_name": "claude-opus-5"},
    )
    meter = Meter()
    meter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))

    assert collect(meter).cache_write == 9000
