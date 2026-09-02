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
