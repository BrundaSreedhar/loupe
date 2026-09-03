from reviewer.nodes.dedupe import group_by_locality
from reviewer.nodes.finalize import finalize
from reviewer.schema import Finding, Verdict


def _f(fid: str, line: int, file: str = "a.py", sev: str = "high", conf: float = 0.9) -> Finding:
    return Finding(
        id=fid,
        produced_by="correctness",
        file=file,
        line=line,
        category="correctness",
        severity=sev,
        summary=f"summary {fid}",
        failure_scenario="concrete failure",
        confidence=conf,
    )


def test_nearby_findings_group_together():
    groups = group_by_locality([_f("a", 10), _f("b", 12), _f("c", 40)])
    assert sorted(len(g) for g in groups) == [1, 2]


def test_same_line_different_files_do_not_group():
    groups = group_by_locality([_f("a", 10), _f("b", 10, file="b.py")])
    assert all(len(g) == 1 for g in groups)


def test_finalize_keeps_only_confirmed():
    state = {
        "merged": [_f("a", 1), _f("b", 2)],
        "verdicts": [
            Verdict(finding_id="a", status="CONFIRMED", reasoning="real"),
            Verdict(finding_id="b", status="REJECTED", reasoning="not real"),
        ],
        "verify": True,
    }
    assert [f.id for f in finalize(state)["accepted"]] == ["a"]


def test_unverified_findings_are_rejected_not_passed_through():
    """A finding whose verification never returned must not survive the gate."""
    state = {"merged": [_f("a", 1)], "verdicts": [], "verify": True}
    assert finalize(state)["accepted"] == []


def test_gate_off_passes_everything():
    state = {"merged": [_f("a", 1), _f("b", 2)], "verdicts": [], "verify": False}
    assert len(finalize(state)["accepted"]) == 2


def test_corrected_line_is_applied():
    state = {
        "merged": [_f("a", 1)],
        "verdicts": [Verdict(finding_id="a", status="CONFIRMED", reasoning="r", corrected_line=42)],
        "verify": True,
    }
    assert finalize(state)["accepted"][0].line == 42


def test_ranking_puts_high_severity_first():
    state = {
        "merged": [_f("low", 1, sev="low", conf=1.0), _f("high", 2, sev="high", conf=0.5)],
        "verdicts": [],
        "verify": False,
    }
    assert [f.id for f in finalize(state)["accepted"]] == ["high", "low"]


def test_both_providers_bind(monkeypatch):
    """Provider selection must produce a working binding for either backend,
    including the structured-output wrapper every reviewer node depends on."""
    import importlib

    from reviewer.schema import FindingBatch

    for provider, key, expected in (
        ("google", "GOOGLE_API_KEY", "ChatGoogleGenerativeAI"),
        ("anthropic", "ANTHROPIC_API_KEY", "ChatAnthropic"),
    ):
        monkeypatch.setenv("REVIEWER_PROVIDER", provider)
        monkeypatch.setenv(key, "dummy-key")
        import reviewer.config as config

        importlib.reload(config)
        llm = config.specialist_llm()
        assert type(llm).__name__ == expected
        assert config.credentials_present()
        llm.with_structured_output(FindingBatch, method="json_schema")

    monkeypatch.delenv("REVIEWER_PROVIDER", raising=False)
    importlib.reload(__import__("reviewer.config", fromlist=["config"]))


def test_report_flags_a_run_that_reviewed_nothing():
    """A pipeline that shows the reviewer nothing scores a confident 0% detection.
    The report has to be able to tell that apart from a cautious reviewer."""
    from evals.scoring import CaseScore, Report

    broken = Report(scores=[
        CaseScore(f"c{i}", "defect", False, 0, 0, 0, 0, 0.0, contexts=0) for i in range(4)
    ])
    working = Report(scores=[
        CaseScore(f"c{i}", "defect", False, 0, 0, 0, 0, 0.0, contexts=2) for i in range(4)
    ])
    assert broken.detection_rate == working.detection_rate == 0.0
    assert broken.reviewed == 0
    assert working.reviewed == 4


def test_empty_report_is_treated_as_failure_not_as_zero():
    """Every case failing produces no scores at all, and every rate becomes 0/0.
    Rendered naively that is a confident 0% — the most misleading output possible."""
    from evals.run_eval import check_did_work
    from evals.scoring import CaseScore, Report

    assert check_did_work({"arm": [Report(scores=[])]}) is False
    assert check_did_work({"arm": [Report(scores=[
        CaseScore("c0", "defect", True, 0, 3, 2, 1, 0.5, contexts=1)
    ])]}) is True


def _finding_with_fix(**fix_kw):
    from reviewer.schema import Finding, FixSuggestion

    base = {"start_line": 3, "end_line": 3, "replacement": "    for i in range(len(xs)):"}
    return Finding(
        id="f1", produced_by="correctness", file="a.py", line=3,
        category="correctness", severity="high", summary="off by one",
        failure_scenario="xs of length 3 indexes 3", confidence=0.9,
        fix=FixSuggestion(**{**base, **fix_kw}),
    )


SRC = "def f(xs):\n    total = 0\n    for i in range(len(xs) + 1):\n        total += xs[i]\n"


def test_a_valid_fix_survives_validation():
    from reviewer.schema import validate_fix

    assert validate_fix(_finding_with_fix(), SRC).fix is not None


def test_a_no_op_fix_is_dropped():
    """A replacement identical to what is already there renders as a live 'apply'
    button that changes nothing — worse than showing no button."""
    from reviewer.schema import validate_fix

    unchanged = "    for i in range(len(xs) + 1):"
    assert validate_fix(_finding_with_fix(replacement=unchanged), SRC).fix is None


def test_a_fix_pointing_outside_the_file_is_dropped():
    from reviewer.schema import validate_fix

    assert validate_fix(_finding_with_fix(start_line=99, end_line=99), SRC).fix is None


def test_a_fix_that_does_not_cover_the_finding_is_dropped():
    """If the replacement range doesn't include the flagged line, the suggestion
    and the comment disagree about what is broken."""
    from reviewer.schema import validate_fix

    assert validate_fix(_finding_with_fix(start_line=1, end_line=1), SRC).fix is None


def test_dropping_a_fix_keeps_the_finding():
    from reviewer.schema import validate_fix

    out = validate_fix(_finding_with_fix(start_line=99, end_line=99), SRC)
    assert out.summary == "off by one" and out.fix is None


def test_github_comment_anchors_to_the_fix_range():
    """GitHub applies a suggestion to exactly the lines the comment spans, so a
    multi-line fix needs start_line as well as line."""
    from reviewer.emit import to_review_comments
    from reviewer.schema import ReviewResult

    single = ReviewResult(request_ref="r", mode="multi", verified=True,
                          accepted=[_finding_with_fix()])
    c = to_review_comments(single)[0]
    assert "```suggestion" in c["body"]
    assert c["line"] == 3 and "start_line" not in c

    multi = ReviewResult(request_ref="r", mode="multi", verified=True,
                         accepted=[_finding_with_fix(start_line=2, end_line=4)])
    c = to_review_comments(multi)[0]
    assert c["start_line"] == 2 and c["line"] == 4 and c["start_side"] == "RIGHT"


def test_nothing_to_review_distinguishes_its_two_causes():
    """An empty diff and a diff that was entirely filtered need different fixes,
    so the message has to say which happened."""
    from typer.testing import CliRunner

    from reviewer.cli import app

    runner = CliRunner()
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for cmd in (["git", "init", "-q", "-b", "main"],
                    ["git", "config", "user.email", "t@t"],
                    ["git", "config", "user.name", "t"]):
            subprocess.run(cmd, cwd=root, check=True, capture_output=True)
        (root / "a.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "one"], cwd=root, check=True, capture_output=True)

        empty = runner.invoke(app, ["local", "HEAD", "--repo-root", str(root)])
        assert "No changes found" in empty.output

        (root / "README.md").write_text("# hi\n")
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "docs"], cwd=root, check=True, capture_output=True)

        filtered = runner.invoke(app, ["local", "HEAD~1", "--repo-root", str(root)])
        assert "none are reviewable" in filtered.output
        assert "README.md" in filtered.output
