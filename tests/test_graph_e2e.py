"""End-to-end wiring test with fake models.

Proves the parts that are easy to get silently wrong and impossible to see in a
unit test: that all four specialists actually run in parallel and all four sets of
findings survive the fan-in reducer, that dedupe merges across reviewers, that the
gate removes rejected findings, and that `single` mode really does run one
reviewer. No network, no key, no spend.
"""

from __future__ import annotations

import re

import pytest

from evals.corpus import build_request
from loupe.schema import (
    FindingBatch,
    IndexedVerdict,
    MergedFinding,
    MergeResult,
    RawFinding,
    VerdictBatch,
)

ORIGINAL = "\n".join(f"value_{i} = compute({i})" for i in range(1, 31)) + "\n"
MUTATED = ORIGINAL.replace("value_4 = compute(4)", "value_4 = compute(5)").replace(
    "value_20 = compute(20)", "value_20 = compute(21)"
)


def _raw(line: int, summary: str, category: str, evidence: str | None = None) -> RawFinding:
    return RawFinding(
        file="sample.py",
        line=line,
        category=category,
        severity="high",
        summary=summary,
        failure_scenario=f"When input reaches line {line}, the wrong value is returned.",
        confidence=0.8,
        # Quote the real line by default: findings that cite code the file does
        # not contain are dropped before dedupe, and every test below would then
        # be measuring the citation check rather than what it means to.
        evidence=MUTATED.splitlines()[line - 1] if evidence is None else evidence,
    )


class _Structured:
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, messages, config=None):
        return self.fn(messages, config)


class _FakeLLM:
    def __init__(self, fn):
        self.fn = fn
        self.calls: list[dict] = []

    def with_structured_output(self, schema, method=None):
        return _Structured(lambda m, c: self._record(m, c))

    def bind(self, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self._record(messages, config)

    def _record(self, messages, config):
        self.calls.append({"config": config or {}, "messages": messages})
        return self.fn(messages, config or {})


@pytest.fixture
def wired(monkeypatch):
    """Patch the factories where the nodes imported them, not at their definition."""

    def specialist_payload(messages, config):
        role = (config.get("metadata") or {}).get("role", "generalist")
        # security+correctness both land on line 4; performance+maintainability on 20.
        line = 4 if role in ("security", "correctness", "generalist") else 20
        return FindingBatch(
            findings=[_raw(line, f"{role} says line {line} is wrong", "correctness")]
        )

    def merge_payload(messages, config):
        # Respect the group actually being merged — run_name is "merge:<file>:<line>".
        line = int(str(config.get("run_name", "merge:x:4")).rsplit(":", 1)[-1])
        return MergeResult(
            findings=[
                MergedFinding(
                    **_raw(line, f"merged defect at {line}", "correctness").model_dump(),
                    covers=[0, 1],
                )
            ]
        )

    def verify_payload(messages, config):
        # Verification is batched per file, so one call may carry several findings.
        # The prompt numbers them "[i] line N"; confirm line 4, reject the rest.
        text = str(messages[-1].content)
        out = []
        for row in text.splitlines():
            m = re.match(r"\[(\d+)\] line (\d+)", row.strip())
            if not m:
                continue
            idx, line = int(m.group(1)), int(m.group(2))
            out.append(
                IndexedVerdict(
                    index=idx,
                    status="CONFIRMED" if line == 4 else "REJECTED",
                    reasoning=f"verdict for line {line}",
                )
            )
        return VerdictBatch(verdicts=out)

    spec = _FakeLLM(specialist_payload)
    merge = _FakeLLM(merge_payload)
    verify = _FakeLLM(verify_payload)

    import loupe.nodes.dedupe as dedupe_mod
    import loupe.nodes.prepare as prepare_mod
    import loupe.nodes.specialists as spec_mod
    import loupe.nodes.verify as verify_mod

    monkeypatch.setattr(spec_mod, "specialist_llm", lambda: spec)
    monkeypatch.setattr(prepare_mod, "specialist_llm", lambda: spec)
    monkeypatch.setattr(dedupe_mod, "merger_llm", lambda: merge)
    monkeypatch.setattr(verify_mod, "verifier_llm", lambda: verify)
    return spec, merge, verify


@pytest.fixture
def request_fixture():
    return build_request("sample.py", ORIGINAL, MUTATED, "e2e")


def test_multi_mode_runs_four_reviewers_and_keeps_all_findings(wired, request_fixture):
    from loupe.runner import run_review

    result = run_review(request_fixture, mode="multi", verify=False)
    roles = {f.produced_by for f in result.raw}
    assert roles == {"security", "correctness", "performance", "maintainability"}
    assert len(result.raw) == 4, "fan-in reducer dropped a branch"


def test_single_mode_runs_one_reviewer(wired, request_fixture):
    from loupe.runner import run_review

    result = run_review(request_fixture, mode="single", verify=False)
    assert len(result.raw) == 1
    assert result.raw[0].produced_by == "generalist"


def test_dedupe_merges_across_reviewers(wired, request_fixture):
    from loupe.runner import run_review

    result = run_review(request_fixture, mode="multi", verify=False)
    # line 4 pair merges into one; line 20 pair merges into one.
    assert len(result.merged) < len(result.raw)


def test_gate_removes_rejected_findings(wired, request_fixture):
    from loupe.runner import run_review

    result = run_review(request_fixture, mode="multi", verify=True)
    assert result.verdicts, "the gate never ran"
    # Assert something survived as well as something being removed: if the gate
    # errors, everything is rejected and an `all(...)` over an empty list passes.
    assert result.accepted, "the gate rejected everything — is it erroring?"
    assert all(f.line == 4 for f in result.accepted), "a rejected finding survived the gate"
    assert 0 < result.usage["rejection_rate"] < 1


def test_verify_spans_carry_finding_ids(wired, request_fixture):
    """Without them the verify fan-out is a set of anonymous sibling spans."""
    _, _, verify = wired
    from loupe.runner import run_review

    run_review(request_fixture, mode="multi", verify=True)
    assert verify.calls
    assert all((c["config"].get("metadata") or {}).get("finding_ids") for c in verify.calls)


def test_warm_cache_skipped_in_single_mode(wired, request_fixture):
    spec, _, _ = wired
    from loupe.runner import run_review

    run_review(request_fixture, mode="single", verify=False)
    assert not any(c["config"].get("run_name") == "warm_cache" for c in spec.calls)


def _warms(spec) -> list:
    return [c for c in spec.calls if c["config"].get("run_name") == "warm_cache"]


def test_warm_cache_skipped_when_prefix_is_too_small(wired, request_fixture):
    """Warming is a blocking call the fan-out waits on. On a small prefix it costs
    a round-trip to save less than one."""
    spec, _, _ = wired
    from loupe.runner import run_review

    run_review(request_fixture, mode="multi", verify=False)
    assert _warms(spec) == []


def test_warm_cache_runs_once_when_prefix_is_large(wired, request_fixture, monkeypatch):
    spec, _, _ = wired
    import loupe.nodes.prepare as prepare_mod

    monkeypatch.setattr(prepare_mod, "WARM_MIN_TOKENS", 0)
    from loupe.runner import run_review

    run_review(request_fixture, mode="multi", verify=False)
    assert len(_warms(spec)) == 1


# ─── phase 4: citations and repository expansion ────────────────────────────


def test_a_finding_quoting_code_that_is_not_there_is_dropped(monkeypatch, request_fixture):
    """The whole pipeline, not the checker in isolation: a reviewer that invents
    its evidence must not reach the report."""
    from loupe.runner import run_review

    def payload(messages, config):
        role = (config.get("metadata") or {}).get("role", "generalist")
        return FindingBatch(
            findings=[
                _raw(4, f"{role} quotes the real line", "correctness"),
                _raw(5, f"{role} quotes a line nobody wrote", "correctness",
                     evidence="value_5 = compute(5) * tax_rate"),
            ]
        )

    spec = _FakeLLM(payload)
    import loupe.nodes.prepare as prepare_mod
    import loupe.nodes.specialists as spec_mod

    monkeypatch.setattr(spec_mod, "specialist_llm", lambda: spec)
    monkeypatch.setattr(prepare_mod, "specialist_llm", lambda: spec)

    result = run_review(request_fixture, mode="single", verify=False)

    assert [f.line for f in result.raw] == [4]
    assert any("quoted" in p.detail for p in result.problems), (
        "a dropped finding has to say so; a quiet drop looks like a clean review"
    )


def test_a_called_definition_reaches_every_reviewer_in_the_shared_prefix(
    wired, monkeypatch, tmp_path
):
    """The point of the index: the reviewer sees the body of what the change
    calls. It has to arrive in the cached block, or four reviewers pay four times
    to read the same definitions."""
    from loupe.runner import run_review

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "validators.py").write_text(
        'def validate(payload):\n    """Reject anything without an id."""\n'
        '    return payload["id"]\n'
    )
    handlers = (
        "from app.validators import validate\n\n\n"
        "def handle(payload):\n    return validate(payload)\n"
    )
    (tmp_path / "app" / "handlers.py").write_text(handlers)

    request = build_request("app/handlers.py", "def handle(payload):\n    pass\n", handlers, "e2e")
    request.repo_root = str(tmp_path)

    spec, _, _ = wired
    run_review(request, mode="multi", verify=False)

    reviewer_calls = [
        c for c in spec.calls if (c["config"].get("metadata") or {}).get("role")
    ]
    assert len(reviewer_calls) == 4
    for call in reviewer_calls:
        # messages are [system, source+references, role rubric]. The definitions
        # belong in the middle one — the part every role shares.
        prefix = str(call["messages"][1].content)
        assert "BEGIN REFERENCED DEFINITIONS" in prefix
        assert "Reject anything without an id." in prefix
        assert "REFERENCED DEFINITIONS" not in str(call["messages"][2].content)
