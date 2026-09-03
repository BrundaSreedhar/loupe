"""Re-judging the close calls.

The point is to spend extra calls only where they change something. A verifier
that confirmed with clear reasoning will confirm again; re-sampling it is pure
cost.
"""

from __future__ import annotations

import pytest

from loupe.consensus import is_borderline, tally, vote_note
from loupe.schema import Finding, Verdict


def _f(confidence: float) -> Finding:
    return Finding(
        id="f1", produced_by="correctness", file="a.py", line=1,
        category="correctness", severity="high", summary="s",
        failure_scenario="concrete", confidence=confidence,
    )


def _v(status: str) -> Verdict:
    return Verdict(finding_id="f1", status=status, reasoning="r")


@pytest.mark.parametrize("confidence,status,expected", [
    (0.3, "CONFIRMED", True),    # reviewer unsure, gate said yes — they disagree
    (0.95, "REJECTED", True),    # reviewer certain, gate said no — they disagree
    (0.95, "CONFIRMED", False),  # both confident it is real
    (0.3, "REJECTED", False),    # both think it is not
])
def test_only_disagreement_is_borderline(confidence, status, expected):
    assert is_borderline(_f(confidence), _v(status)) is expected


def test_majority_confirms():
    assert tally([_v("CONFIRMED"), _v("CONFIRMED"), _v("REJECTED")]) == ("CONFIRMED", 2, 3)


def test_majority_rejects():
    assert tally([_v("REJECTED"), _v("REJECTED"), _v("CONFIRMED")]) == ("REJECTED", 2, 3)


def test_a_tie_rejects():
    """An even split is not confidence, and a false positive is the thing this
    stage exists to prevent."""
    status, _, _ = tally([_v("CONFIRMED"), _v("REJECTED")])
    assert status == "REJECTED"


def test_no_votes_rejects():
    assert tally([])[0] == "REJECTED"


def test_vote_note_distinguishes_unanimity_from_a_squeaker():
    assert vote_note(3, 3) != vote_note(2, 3)
    assert "3 of 3" in vote_note(3, 3)


def test_disabled_by_default(isolated_config):
    """Extra samples cost calls, so they are opt-in."""
    assert isolated_config().CONSENSUS_SAMPLES == 1


def test_routing_skips_everything_when_disabled(monkeypatch, isolated_config):
    monkeypatch.setenv("LOUPE_CONSENSUS_SAMPLES", "1")
    isolated_config()
    import importlib

    import loupe.graph as graph_mod
    importlib.reload(graph_mod)
    assert graph_mod.route_consensus({"merged": [_f(0.3)], "verdicts": [_v("CONFIRMED")]}) \
        == "finalize"


def test_routing_sends_only_the_close_calls(monkeypatch, isolated_config):
    monkeypatch.setenv("LOUPE_CONSENSUS_SAMPLES", "3")
    isolated_config()
    import importlib

    import loupe.graph as graph_mod
    importlib.reload(graph_mod)

    from loupe.schema import FileDiff, ReviewRequest

    clear = _f(0.95)
    close = _f(0.2)
    close = close.model_copy(update={"id": "f2"})
    state = {
        "merged": [clear, close],
        "verdicts": [
            Verdict(finding_id="f1", status="CONFIRMED", reasoning="r"),
            Verdict(finding_id="f2", status="CONFIRMED", reasoning="r"),
        ],
        "request": ReviewRequest(
            source="local", ref="r",
            files=[FileDiff(path="a.py", content_after="x = 1\n")],
        ),
    }
    sends = graph_mod.route_consensus(state)
    assert len(sends) == 1, "only the finding whose signals disagree should be re-judged"
    assert sends[0].arg["finding"].id == "f2"

    monkeypatch.delenv("LOUPE_CONSENSUS_SAMPLES")
    isolated_config()
    importlib.reload(graph_mod)


def test_consensus_verdict_overrides_the_first_pass():
    """The whole mechanism is worthless if finalize keeps the original verdict."""
    from loupe.nodes.finalize import finalize

    state = {
        "merged": [_f(0.3)],
        "verdicts": [Verdict(finding_id="f1", status="CONFIRMED", reasoning="first pass")],
        "consensus": [Verdict(finding_id="f1", status="REJECTED", reasoning="2 of 3 disagreed")],
        "verify": True,
    }
    assert finalize(state)["accepted"] == [], "the re-judged verdict must win"

    state["consensus"] = [Verdict(finding_id="f1", status="CONFIRMED", reasoning="3 of 3")]
    assert len(finalize(state)["accepted"]) == 1


def test_a_finding_never_re_judged_keeps_its_original_verdict():
    from loupe.nodes.finalize import finalize

    state = {
        "merged": [_f(0.9)],
        "verdicts": [Verdict(finding_id="f1", status="CONFIRMED", reasoning="clear")],
        "consensus": [],
        "verify": True,
    }
    assert len(finalize(state)["accepted"]) == 1
