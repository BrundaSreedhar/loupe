"""End-to-end wiring test with fake models.

Proves the parts that are easy to get silently wrong and impossible to see in a
unit test: that all four specialists actually run in parallel and all four sets of
findings survive the fan-in reducer, that dedupe merges across reviewers, that the
gate removes rejected findings, and that `single` mode really does run one
reviewer. No network, no key, no spend.
"""

from __future__ import annotations

import pytest

from evals.corpus import build_request
from reviewer.schema import FindingBatch, MergedFinding, MergeResult, RawFinding, VerdictDecision

ORIGINAL = "\n".join(f"value_{i} = compute({i})" for i in range(1, 31)) + "\n"
MUTATED = ORIGINAL.replace("value_4 = compute(4)", "value_4 = compute(5)").replace(
    "value_20 = compute(20)", "value_20 = compute(21)"
)


def _raw(line: int, summary: str, category: str) -> RawFinding:
    return RawFinding(
        file="sample.py",
        line=line,
        category=category,
        severity="high",
        summary=summary,
        failure_scenario=f"When input reaches line {line}, the wrong value is returned.",
        confidence=0.8,
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
        self.calls.append({"config": config or {}})
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
        fid = (config.get("metadata") or {}).get("finding_id", "")
        # Confirm the line-4 defect, reject the line-20 one. run_name is
        # "verify:<file>:<line>", which is unambiguous where prose is not.
        line = int(str(config.get("run_name", "verify:x:0")).rsplit(":", 1)[-1])
        status = "CONFIRMED" if line == 4 else "REJECTED"
        return VerdictDecision(status=status, reasoning=f"verdict for {fid}")

    spec = _FakeLLM(specialist_payload)
    merge = _FakeLLM(merge_payload)
    verify = _FakeLLM(verify_payload)

    import reviewer.nodes.dedupe as dedupe_mod
    import reviewer.nodes.prepare as prepare_mod
    import reviewer.nodes.specialists as spec_mod
    import reviewer.nodes.verify as verify_mod

    monkeypatch.setattr(spec_mod, "specialist_llm", lambda: spec)
    monkeypatch.setattr(prepare_mod, "specialist_llm", lambda: spec)
    monkeypatch.setattr(dedupe_mod, "merger_llm", lambda: merge)
    monkeypatch.setattr(verify_mod, "verifier_llm", lambda: verify)
    return spec, merge, verify


@pytest.fixture
def request_fixture():
    return build_request("sample.py", ORIGINAL, MUTATED, "e2e")


def test_multi_mode_runs_four_reviewers_and_keeps_all_findings(wired, request_fixture):
    from reviewer.runner import run_review

    result = run_review(request_fixture, mode="multi", verify=False)
    roles = {f.produced_by for f in result.raw}
    assert roles == {"security", "correctness", "performance", "maintainability"}
    assert len(result.raw) == 4, "fan-in reducer dropped a branch"


def test_single_mode_runs_one_reviewer(wired, request_fixture):
    from reviewer.runner import run_review

    result = run_review(request_fixture, mode="single", verify=False)
    assert len(result.raw) == 1
    assert result.raw[0].produced_by == "generalist"


def test_dedupe_merges_across_reviewers(wired, request_fixture):
    from reviewer.runner import run_review

    result = run_review(request_fixture, mode="multi", verify=False)
    # line 4 pair merges into one; line 20 pair merges into one.
    assert len(result.merged) < len(result.raw)


def test_gate_removes_rejected_findings(wired, request_fixture):
    from reviewer.runner import run_review

    result = run_review(request_fixture, mode="multi", verify=True)
    assert result.verdicts, "the gate never ran"
    assert all(f.line == 4 for f in result.accepted), "a rejected finding survived the gate"
    assert result.usage["rejection_rate"] > 0


def test_verify_spans_carry_finding_id(wired, request_fixture):
    """Without finding_id the verify fan-out is N anonymous sibling spans."""
    _, _, verify = wired
    from reviewer.runner import run_review

    run_review(request_fixture, mode="multi", verify=True)
    assert all("finding_id" in (c["config"].get("metadata") or {}) for c in verify.calls)


def test_warm_cache_skipped_in_single_mode(wired, request_fixture):
    spec, _, _ = wired
    from reviewer.runner import run_review

    run_review(request_fixture, mode="single", verify=False)
    assert not any(c["config"].get("run_name") == "warm_cache" for c in spec.calls)


def test_warm_cache_runs_once_in_multi_mode(wired, request_fixture):
    spec, _, _ = wired
    from reviewer.runner import run_review

    run_review(request_fixture, mode="multi", verify=False)
    warms = [c for c in spec.calls if c["config"].get("run_name") == "warm_cache"]
    assert len(warms) == 1
