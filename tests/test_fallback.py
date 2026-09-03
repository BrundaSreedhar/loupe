"""Falling back to a local model, and only for the right reason.

A daily exhaustion is worth swapping a frontier model for a 7B one. A seven-second
per-minute limit is not — that should wait and retry on the good model.
"""

from __future__ import annotations

import pytest

from loupe.fallback import LocalFallback, reset_usage, usage
from tests.test_quota import DAILY, PER_MINUTE


class _Rate(Exception):
    code = 429


class _Primary:
    def __init__(self, exc=None):
        self.calls = 0
        self.exc = exc

    def invoke(self, x, config=None, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        return "primary"


class _Local:
    def __init__(self):
        self.calls = 0

    def invoke(self, x, config=None, **kw):
        self.calls += 1
        return "local"


@pytest.fixture(autouse=True)
def _clean():
    reset_usage()
    yield
    reset_usage()


def test_daily_quota_falls_back():
    primary, local = _Primary(_Rate(DAILY)), _Local()
    assert LocalFallback(primary, local, "verify").invoke("x") == "local"
    assert local.calls == 1
    assert usage() == {"verify"}


def test_per_minute_limit_does_not_fall_back():
    """Swapping models over a seven-second hiccup silently downgrades the review."""
    primary, local = _Primary(_Rate(PER_MINUTE)), _Local()
    with pytest.raises(_Rate):
        LocalFallback(primary, local, "verify").invoke("x")
    assert local.calls == 0
    assert usage() == set()


def test_other_errors_are_not_swallowed():
    primary, local = _Primary(ValueError("bad schema")), _Local()
    with pytest.raises(ValueError):
        LocalFallback(primary, local, "verify").invoke("x")
    assert local.calls == 0


def test_no_fallback_when_primary_succeeds():
    primary, local = _Primary(), _Local()
    assert LocalFallback(primary, local, "verify").invoke("x") == "primary"
    assert local.calls == 0 and usage() == set()


def test_fallback_is_reported_as_an_error_not_a_footnote(monkeypatch):
    """A review answered by a small local model is a different artefact. If that
    is not on the output, the numbers from it look like the real thing."""
    import loupe.fallback as fb
    import loupe.runner as runner

    monkeypatch.setattr(runner, "usage", lambda: {"verify", "specialist:security"})

    class _Graph:
        def invoke(self, state, config=None):
            return {"findings": [], "merged": [], "verdicts": [], "accepted": [],
                    "problems": [], "contexts": {}}

    monkeypatch.setattr(runner, "_graph", lambda: _Graph())
    from loupe.schema import ReviewRequest

    result = runner.run_review(ReviewRequest(source="local", ref="r"))
    fallback = [p for p in result.problems if p.stage == "fallback"]
    assert fallback and fallback[0].severity == "error"
    assert "not comparable" in fallback[0].detail
    assert result.usage["fallback_stages"] == 2
    assert fb  # imported to make the dependency explicit


def test_model_provider_mismatch_is_corrected(monkeypatch, isolated_config):
    """LOUPE_MODEL left set to a Gemini model while the provider is ollama
    otherwise asks Ollama for 'gemini-3.5-flash' and fails much later."""
    monkeypatch.setenv("LOUPE_PROVIDER", "ollama")
    monkeypatch.setenv("LOUPE_MODEL", "gemini-3.5-flash")
    assert isolated_config().MODEL == "qwen2:7b"

    monkeypatch.setenv("LOUPE_MODEL", "qwen2:7b-32k")
    assert isolated_config().MODEL == "qwen2:7b-32k"


def test_verifier_can_use_a_different_model_from_the_reviewers(monkeypatch, isolated_config):
    """Google's daily cap is per project AND per model, so pointing roles at
    different models gives each its own allowance instead of sharing one."""
    monkeypatch.setenv("LOUPE_PROVIDER", "google")
    monkeypatch.setenv("LOUPE_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LOUPE_VERIFIER_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("LOUPE_CHEAP_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    config = isolated_config()
    assert config.MODEL == "gemini-3.5-flash"
    assert config.VERIFIER_MODEL == "gemini-3.6-flash"
    assert config.specialist_llm().model.endswith("gemini-3.5-flash")
    assert config.verifier_llm().model.endswith("gemini-3.6-flash")
    assert config.merger_llm().model.endswith("gemini-3.5-flash-lite")


def test_verifier_model_defaults_to_the_main_model(monkeypatch, isolated_config):
    """Must not depend on whether the person running the tests has a verifier
    model configured in their own ~/.config/loupe/.env."""
    monkeypatch.setenv("LOUPE_PROVIDER", "google")
    monkeypatch.setenv("LOUPE_MODEL", "gemini-3.5-flash")
    monkeypatch.delenv("LOUPE_VERIFIER_MODEL", raising=False)

    config = isolated_config()
    assert config.VERIFIER_MODEL == config.MODEL == "gemini-3.5-flash"
