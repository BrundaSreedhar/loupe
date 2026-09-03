"""Telling a per-minute 429 from a per-day one.

The fixture is the real error body this project received, not an invented one —
including the detail that makes naive handling fail: Google sends a short
`retryDelay` even when the exhausted window is the day.
"""

from __future__ import annotations

import pytest

from agentgate.quota import (
    DailyQuotaExhausted,
    classify,
    is_rate_limit,
    raise_if_terminal,
    should_retry,
)

DAILY = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', "
    "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', "
    "'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
    "'retryDelay': '51s'}]}}"
)

PER_MINUTE = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '10'}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '7s'}]}}"
)


class _Err(Exception):
    code = 429


def test_daily_exhaustion_is_recognised():
    info = classify(_Err(DAILY))
    assert info is not None
    assert info.scope == "per_day"
    assert info.quota_value == 20
    assert info.retryable is False


def test_daily_retry_delay_is_not_treated_as_the_wait():
    """Google sends retryDelay: 51s even for a daily cap. Honouring it means
    retrying every 51 seconds, all day, forever."""
    info = classify(_Err(DAILY))
    assert info.retry_after == 51.0
    assert info.retryable is False, "51s is not when the daily window reopens"


def test_per_minute_is_retryable():
    info = classify(_Err(PER_MINUTE))
    assert info.scope == "per_minute"
    assert info.retryable is True
    assert info.retry_after == 7.0


def test_unknown_rate_limit_is_retried():
    """Retrying a transient limit needlessly costs seconds; refusing a transient
    limit costs the run. Bias toward retrying when the body is unparseable."""
    info = classify(_Err("429 RESOURCE_EXHAUSTED, no further detail"))
    assert info.scope == "unknown"
    assert info.retryable is True


def test_non_rate_limit_errors_are_not_classified():
    assert classify(ValueError("bad schema")) is None
    assert is_rate_limit(ValueError("bad schema")) is False


def test_should_retry_matches_the_policy():
    assert should_retry(_Err(PER_MINUTE)) is True
    assert should_retry(_Err(DAILY)) is False
    # A 503 from a busy model is exactly what retries are for.
    assert should_retry(RuntimeError("503 UNAVAILABLE, high demand")) is True


def test_raise_if_terminal_only_fires_on_daily():
    raise_if_terminal(_Err(PER_MINUTE))          # must not raise
    raise_if_terminal(ValueError("unrelated"))   # must not raise
    with pytest.raises(DailyQuotaExhausted) as exc:
        raise_if_terminal(_Err(DAILY))
    assert "20/day" in str(exc.value)


def test_verify_propagates_quota_instead_of_rejecting_everything():
    """A quota failure inside the gate is not evidence about the findings.
    Silently rejecting them produces a clean review that never happened."""
    import agentgate.nodes.verify as verify_mod
    from agentgate.schema import Finding

    class _Boom:
        def with_structured_output(self, *a, **k):
            return self

        def invoke(self, *a, **k):
            raise _Err(DAILY)

    original = verify_mod.verifier_llm
    verify_mod.verifier_llm = lambda: _Boom()
    try:
        finding = Finding(
            id="x", produced_by="correctness", file="a.py", line=1,
            category="correctness", severity="high", summary="s",
            failure_scenario="f", confidence=0.9,
        )
        with pytest.raises(DailyQuotaExhausted):
            verify_mod.verify({"path": "a.py", "findings": [finding], "source": "x = 1"})
    finally:
        verify_mod.verifier_llm = original
