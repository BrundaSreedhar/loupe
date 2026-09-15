"""Scoring, and specifically the gap between the two detection columns.

`detection_rate` asks only whether something was flagged near the seeded line. That
is the right default — a reviewer can be correct about a defect and describe it
differently from the mutator — but on its own it counts a coincidence as a catch.
`strict_detection_rate` adds the category, and the distance between them is how
much of the headline number is luck.
"""

from __future__ import annotations

from evals.corpus import Case
from evals.mutations import Mutation
from evals.scoring import Report, score_case
from loupe.schema import FileDiff, Finding, ReviewRequest, ReviewResult, TokenUsage

PATH = "pkg/charge.py"


def _case(kind: str = "defect", category: str = "correctness", line: int = 20) -> Case:
    request = ReviewRequest(
        source="local",
        ref="test",
        files=[FileDiff(path=PATH, change_type="modified", content_after="x = 1\n")],
    )
    truth = (
        Mutation(name="m", category=category, line=line, before="a", after="b", description="d")
        if kind == "defect"
        else None
    )
    return Case(id=f"{kind}-000-m", kind=kind, request=request, truth=truth)


def _finding(line: int, category: str = "correctness", path: str = PATH) -> Finding:
    return Finding(
        id="f1",
        produced_by="correctness",
        file=path,
        line=line,
        category=category,
        severity="high",
        summary="s",
        failure_scenario="concrete inputs produce the wrong total",
        confidence=0.9,
    )


def _result(*findings: Finding) -> ReviewResult:
    return ReviewResult(
        request_ref="test",
        mode="multi",
        verified=True,
        accepted=list(findings),
        usage={"raw_count": len(findings), "merged_count": len(findings)},
        tokens=TokenUsage(calls=1),
    )


def test_the_right_line_for_the_right_reason_counts_for_both():
    s = score_case(_case(category="correctness"), _result(_finding(20, "correctness")))
    assert s.detected is True
    assert s.detected_category is True


def test_the_right_line_for_the_wrong_reason_counts_only_for_the_loose_column():
    """The case this exists for: a maintainability reviewer objecting to a name on
    the same line as a seeded logic bug. Worth counting, not worth counting as a
    catch of that defect."""
    s = score_case(_case(category="correctness"), _result(_finding(20, "maintainability")))
    assert s.detected is True
    assert s.detected_category is False


def test_a_finding_within_tolerance_still_counts():
    """Reviewers are routinely a line or two out, and the harness allows three."""
    s = score_case(_case(line=20), _result(_finding(22)))
    assert s.detected is True
    assert s.detected_category is True


def test_a_finding_outside_tolerance_counts_for_neither():
    s = score_case(_case(line=20), _result(_finding(40)))
    assert s.detected is False
    assert s.detected_category is False


def test_a_finding_in_another_file_counts_for_neither():
    s = score_case(_case(line=20), _result(_finding(20, path="pkg/other.py")))
    assert s.detected is False
    assert s.detected_category is False


def test_one_right_reason_among_several_findings_is_enough():
    """Two reviewers flag the same line, one of them for the seeded reason. That is
    a catch — the panel found it, which is what the panel is for."""
    s = score_case(
        _case(category="security"),
        _result(_finding(20, "maintainability"), _finding(20, "security")),
    )
    assert s.detected_category is True


def test_clean_cases_score_neither_column():
    s = score_case(_case(kind="clean"), _result(_finding(20)))
    assert s.detected is None
    assert s.detected_category is None
    assert s.false_positives == 1


def test_the_report_separates_the_two_rates():
    report = Report(
        scores=[
            score_case(_case(category="correctness"), _result(_finding(20, "correctness"))),
            score_case(_case(category="correctness"), _result(_finding(20, "maintainability"))),
            score_case(_case(category="correctness"), _result(_finding(90))),
        ]
    )
    assert report.detection_rate == 2 / 3
    assert report.strict_detection_rate == 1 / 3
    assert report.as_dict()["strict_detection_rate"] == 1 / 3


def test_strict_detection_is_never_higher_than_loose():
    """A structural property: the strict rule is the loose rule plus a condition, so
    a run where strict exceeds loose means one of them is computed wrong."""
    report = Report(
        scores=[
            score_case(_case(category=c), _result(_finding(20, f)))
            for c in ("correctness", "security")
            for f in ("correctness", "security", "maintainability")
        ]
    )
    assert report.strict_detection_rate <= report.detection_rate
