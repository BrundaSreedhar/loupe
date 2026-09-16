"""Reductions in model calls per review.

Two kinds here. The first two cost nothing — they remove calls whose answers were
being discarded. The rest trade something, and belong behind a flag the harness
can price.
"""

from __future__ import annotations

import pytest

from loupe.schema import Finding


class _FakeMerger:
    """Stands in for the merger binding, so no test here needs a provider key.

    `dedupe` builds its merger before it knows whether any group is worth
    merging, so even a test with nothing to merge reaches real client
    construction — which fails outright when no key is set. `calls` records
    whatever did get through.
    """

    def __init__(self, fails_with: Exception | None = None):
        self.calls: list[tuple] = []
        self._fails_with = fails_with

    def with_structured_output(self, *a, **k):
        return self

    def invoke(self, *a, **k):
        self.calls.append((a, k))
        raise self._fails_with or AssertionError("no merge call was expected here")


def _f(fid: str, line: int, sev: str = "high", conf: float = 0.9, by: str = "correctness"):
    return Finding(
        id=fid, produced_by=by, file="a.py", line=line, category="correctness",
        severity=sev, summary=f"s{fid}", failure_scenario="concrete", confidence=conf,
    )


def test_findings_that_ranking_would_discard_are_not_verified(monkeypatch, isolated_config):
    """Verification costs a call per file. Paying to judge a finding that ranking
    then drops unseen is the definition of a wasted call."""
    monkeypatch.setenv("LOUPE_MAX_REPORTED", "2")
    monkeypatch.setenv("LOUPE_VERIFY_HEADROOM", "2")
    config = isolated_config()
    import loupe.nodes.dedupe as dedupe_mod

    # After the reload, not before: reloading the module rebinds `merger_llm`.
    merger = _FakeMerger()
    monkeypatch.setattr(dedupe_mod, "merger_llm", lambda: merger)

    many = [_f(f"f{i}", line=i * 100, sev="low", conf=0.5) for i in range(20)]
    out = dedupe_mod.dedupe({"findings": many})
    assert len(out["merged"]) == config.MAX_REPORTED * config.VERIFY_HEADROOM == 4
    assert merger.calls == [], "20 findings 100 lines apart have nothing to merge"


def test_the_shortlist_keeps_the_highest_ranked(monkeypatch, isolated_config):
    monkeypatch.setenv("LOUPE_MAX_REPORTED", "1")
    monkeypatch.setenv("LOUPE_VERIFY_HEADROOM", "1")
    isolated_config()
    import loupe.nodes.dedupe as dedupe_mod

    merger = _FakeMerger()
    monkeypatch.setattr(dedupe_mod, "merger_llm", lambda: merger)

    out = dedupe_mod.dedupe({"findings": [
        _f("low", 100, sev="low", conf=0.4),
        _f("high", 200, sev="high", conf=0.95),
    ]})
    assert [f.id for f in out["merged"]] == ["high"]
    assert merger.calls == []


def test_one_reviewer_filing_twice_costs_no_merge_call(monkeypatch):
    """A single reviewer that reported two nearby lines saw both at once — it is
    describing two things, not the same thing twice."""
    import loupe.nodes.dedupe as dedupe_mod

    merger = _FakeMerger()
    monkeypatch.setattr(dedupe_mod, "merger_llm", lambda: merger)
    out = dedupe_mod.dedupe({"findings": [
        _f("a", 10, by="security"), _f("b", 11, by="security"),
    ]})
    assert len(out["merged"]) == 2
    assert merger.calls == [], "no model call should merge one reviewer's own findings"


def test_two_reviewers_at_the_same_spot_still_get_merged(monkeypatch):
    """The saving must not swallow the case dedupe exists for."""
    import loupe.nodes.dedupe as dedupe_mod

    merger = _FakeMerger(RuntimeError("merge unavailable"))
    monkeypatch.setattr(dedupe_mod, "merger_llm", lambda: merger)
    dedupe_mod.dedupe({"findings": [
        _f("a", 10, by="security"), _f("b", 11, by="correctness"),
    ]})
    assert len(merger.calls) == 1, (
        "two different reviewers at one spot is exactly what merging is for"
    )


@pytest.mark.parametrize("fanout,roles,expected", [
    ("parallel", "security,correctness,performance,maintainability", 4),
    ("parallel", "security,correctness", 2),
    ("combined", "security,correctness,performance,maintainability", 1),
])
def test_reviewer_call_count_is_configurable(fanout, roles, expected, monkeypatch, isolated_config):
    monkeypatch.setenv("LOUPE_FANOUT", fanout)
    monkeypatch.setenv("LOUPE_ROLES", roles)
    isolated_config()
    import loupe.graph as graph_mod

    sends = graph_mod.fan_out({"contexts": {"a.py": object()}, "mode": "multi", "request": None})
    assert len(sends) == expected


def test_combined_rubric_carries_every_specialist_s_checklist():
    """One call instead of four is only acceptable if nothing is dropped from what
    the reviewer is asked to look for."""
    from loupe.prompts.rubrics import RUBRICS, combined_rubric

    looks = combined_rubric()["looks_for"]
    for role in ("security", "correctness", "performance", "maintainability"):
        first_line = RUBRICS[role]["looks_for"].splitlines()[0]
        assert first_line in looks, f"{role} checklist missing from the combined prompt"


def test_an_unknown_role_fails_loudly(monkeypatch, isolated_config):
    monkeypatch.setenv("LOUPE_ROLES", "security,typos")
    with pytest.raises(ValueError, match="typos"):
        isolated_config()
