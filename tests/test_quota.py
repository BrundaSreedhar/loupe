"""Telling a per-minute 429 from a per-day one.

The fixture is the real error body this project received, not an invented one —
including the detail that makes naive handling fail: Google sends a short
`retryDelay` even when the exhausted window is the day.
"""

from __future__ import annotations

import pytest

from loupe.quota import (
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
    import loupe.nodes.verify as verify_mod
    from loupe.schema import Finding

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


def test_a_daily_cap_from_a_reviewer_reaches_the_caller_as_a_stop_signal(monkeypatch):
    """Observed on a real run. The nodes with broad excepts translate a daily cap
    already; a reviewer branch has none, because it has no failure to swallow. So
    an exhausted quota came out as the provider's own error, the eval harness read
    it as one bad case rather than a dead run, and carried on through the corpus
    failing identically — the exact waste quota.py exists to prevent."""
    import loupe.runner as runner
    from loupe.quota import DailyQuotaExhausted
    from loupe.schema import ReviewRequest

    def blows_up(*args, **kwargs):
        raise _Err(DAILY)

    monkeypatch.setattr(runner, "_invoke", blows_up)
    with pytest.raises(DailyQuotaExhausted):
        runner.run_review(ReviewRequest(source="local", ref="HEAD"))


def test_an_ordinary_failure_is_not_dressed_up_as_a_quota_problem(monkeypatch):
    import loupe.runner as runner
    from loupe.schema import ReviewRequest

    def blows_up(*args, **kwargs):
        raise ValueError("something else went wrong")

    monkeypatch.setattr(runner, "_invoke", blows_up)
    with pytest.raises(ValueError):
        runner.run_review(ReviewRequest(source="local", ref="HEAD"))


# ─── a rejected credential ──────────────────────────────────────────────────


class _Anthropic401(Exception):
    """The shape langchain_anthropic raises: a status_code and the body in str()."""

    status_code = 401

    def __str__(self) -> str:
        return (
            "Error code: 401 - {'type': 'error', 'error': {'type': "
            "'authentication_error', 'message': 'API key is invalid.'}, "
            "'request_id': None}"
        )


def test_a_401_is_terminal_not_retryable():
    from loupe.quota import AuthenticationFailed, is_auth_error, raise_if_terminal, should_retry

    exc = _Anthropic401()
    assert is_auth_error(exc)
    # Retrying a rejected key is strictly waste, exactly like a daily cap.
    assert should_retry(exc) is False
    with pytest.raises(AuthenticationFailed):
        raise_if_terminal(exc)


def test_google_phrasings_are_recognised_too():
    from loupe.quota import is_auth_error

    for text in ("API_KEY_INVALID", "UNAUTHENTICATED", "invalid_api_key"):
        assert is_auth_error(Exception(text)), text


def test_an_ordinary_failure_is_not_an_auth_error():
    """The guard must not turn a transient 503 into a terminal one — that would
    abort runs that a retry would have saved."""
    from loupe.quota import is_auth_error, raise_if_terminal, should_retry

    exc = Exception("Error code: 503 - model overloaded")
    assert not is_auth_error(exc)
    assert should_retry(exc) is True
    raise_if_terminal(exc)  # does not raise


def _warm_state():
    """Enough state to get past warm_cache's own guards: it skips on a single
    reviewer and on a prefix too small to be worth a round trip."""
    ctx = type("Ctx", (), {"tokens": 10_000})()
    return {
        "contexts": {"a.py": ctx},
        "references": [],
        "mode": "multi",
        "request": None,
    }


def test_a_failed_warm_on_a_bad_key_stops_the_review(monkeypatch):
    """The complaint this answers: the warm 401'd, the handler logged 'reviewers
    will run cold', and four reviewers then failed the same way — producing an
    empty review that reads like a clean one."""
    from loupe.nodes import prepare
    from loupe.quota import AuthenticationFailed

    def boom(*a, **k):
        raise _Anthropic401()

    monkeypatch.setattr(prepare, "specialist_llm", boom)

    with pytest.raises(AuthenticationFailed):
        prepare.warm_cache(_warm_state(), {})


def test_a_warm_that_fails_for_an_ordinary_reason_still_only_warns(monkeypatch):
    """The original contract has to survive: a timeout costs money, not the run."""
    from loupe.nodes import prepare

    def boom(*a, **k):
        raise TimeoutError("took too long")

    monkeypatch.setattr(prepare, "specialist_llm", boom)

    out = prepare.warm_cache(_warm_state(), {})
    assert out["problems"] and out["problems"][0].stage == "warm_cache"


def test_a_translated_terminal_error_is_not_retried():
    """A node that catches a raw 401 and re-raises AuthenticationFailed has made
    a decision. The retry policy reads wording, and the wrapper does not carry the
    provider's — so without a type check it retried the node twice more."""
    from loupe.quota import AuthenticationFailed, DailyQuotaExhausted, should_retry

    assert should_retry(AuthenticationFailed("the provider rejected the credential")) is False
    assert should_retry(DailyQuotaExhausted("daily quota exhausted")) is False
