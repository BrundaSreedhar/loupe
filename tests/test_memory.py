"""Finding identity across runs, and reporting only what changed."""

from __future__ import annotations

import pytest

from loupe.fingerprint import compute
from loupe.memory import diff, load, path_for, save
from loupe.schema import Finding

SRC = (
    "def average(xs):\n"
    "    total = 0\n"
    "    for i in range(len(xs) + 1):\n"
    "        total += xs[i]\n"
    "    return total / len(xs)\n"
)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LOUPE_STATE_DIR", str(tmp_path / "state"))


def _f(fp: str, fid: str = "x") -> Finding:
    return Finding(
        id=fid, produced_by="correctness", file="a.py", line=3,
        category="correctness", severity="high", summary="off by one",
        failure_scenario="concrete", confidence=0.9, fingerprint=fp,
    )


# ─── identity ───────────────────────────────────────────────────────────────


def test_survives_lines_moving():
    """The whole point. An edit above a defect must not make it a new finding."""
    shifted = "# added\n# lines\n" + SRC
    assert compute("a.py", "correctness", SRC, 3) == compute("a.py", "correctness", shifted, 5)


def test_survives_reformatting():
    reflowed = SRC.replace("range(len(xs) + 1)", "range( len(xs)+1 )")
    assert compute("a.py", "correctness", SRC, 3) == compute("a.py", "correctness", reflowed, 3)


def test_changes_when_the_defect_is_fixed():
    fixed = SRC.replace("len(xs) + 1", "len(xs)")
    assert compute("a.py", "correctness", SRC, 3) != compute("a.py", "correctness", fixed, 3)


def test_two_categories_on_one_line_are_two_findings():
    """A query can be both injectable and inside a loop. Fixing one must not
    silence the other."""
    assert compute("a.py", "security", SRC, 3) != compute("a.py", "performance", SRC, 3)


def test_same_code_in_different_files_differs():
    assert compute("a.py", "correctness", SRC, 3) != compute("b.py", "correctness", SRC, 3)


def test_identical_lines_elsewhere_in_a_file_do_not_collide():
    """A one-line window would collide on repeated lines; the surrounding context
    is what separates them."""
    src = "x = 1\ntotal = 0\ny = 2\n\nz = 3\ntotal = 0\nw = 4\n"
    assert compute("a.py", "correctness", src, 2) != compute("a.py", "correctness", src, 6)


# ─── memory ─────────────────────────────────────────────────────────────────


def test_round_trip():
    save("repo", "main", [_f("fp1")])
    assert "fp1" in load("repo", "main")


def test_branches_are_separate():
    save("repo", "main", [_f("fp1")])
    assert load("repo", "other") == {}


def test_unreadable_state_is_treated_as_empty():
    """Corrupt memory costs a repeated comment, never a failed review."""
    p = path_for("repo", "main")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json")
    assert load("repo", "main") == {}


def test_a_findings_only_run_reports_everything_as_new():
    d = diff({}, [_f("fp1"), _f("fp2", "y")], first_run=True)
    assert len(d.new) == 2 and d.persisting == [] and d.resolved == []


def test_classifies_new_persisting_and_resolved():
    previous = {"stays": {}, "gone": {}}
    d = diff(previous, [_f("stays"), _f("arrived", "y")], first_run=False)
    assert [f.fingerprint for f in d.new] == ["arrived"]
    assert [f.fingerprint for f in d.persisting] == ["stays"]
    assert d.resolved == ["gone"]


def test_a_finding_without_a_fingerprint_is_always_new():
    """No source to hash means no identity — it must never be silently suppressed
    as something already reported."""
    d = diff({"": {}}, [_f("")], first_run=False)
    assert d.new == [] and d.persisting == []


def test_state_is_not_written_into_the_repository(tmp_path):
    """A machine-generated file in the working tree would end up in the next diff,
    and the reviewer would be asked to review it."""
    save("repo", "main", [_f("fp1")])
    assert str(path_for("repo", "main")).startswith(str(tmp_path))


# ─── the behaviour that decides whether anyone leaves it switched on ─────────


def _run(findings, request, remember=True):
    """Drive finalize directly: it is where memory is consulted."""
    from loupe.nodes.finalize import finalize

    return finalize({
        "merged": findings, "verdicts": [], "verify": False,
        "request": request, "remember": remember,
    })


def _request(tmp_path):
    from loupe.schema import ReviewRequest

    return ReviewRequest(source="local", ref="HEAD~1", repo_root=str(tmp_path))


def test_reviewing_an_unchanged_branch_reports_nothing_new(tmp_path):
    request = _request(tmp_path)
    findings = [_f("fp1"), _f("fp2", "y")]

    first = _run(findings, request)
    assert first["delta"].first_run is True
    assert len(first["delta"].new) == 2

    second = _run(findings, request)
    assert second["delta"].new == [], "nothing changed, so nothing is new"
    assert len(second["delta"].persisting) == 2
    # The findings themselves are still returned — suppression is a display
    # decision, not a reason to lose data.
    assert len(second["accepted"]) == 2


def test_a_fixed_finding_is_reported_as_resolved(tmp_path):
    request = _request(tmp_path)
    _run([_f("fp1"), _f("fp2", "y")], request)
    after = _run([_f("fp1")], request)
    assert after["delta"].resolved == ["fp2"]


def test_fresh_ignores_memory(tmp_path):
    request = _request(tmp_path)
    _run([_f("fp1")], request)
    again = _run([_f("fp1")], request, remember=False)
    assert "delta" not in again, "--fresh must not consult or update memory"


def test_the_delta_is_rendered_not_the_whole_list_again(tmp_path):
    import io

    from rich.console import Console

    from loupe.emit import render
    from loupe.schema import ReviewResult

    class _Delta:
        first_run = False
        resolved = ["gone"]
        persisting = [_f("stays")]
        new: list = []

    console = Console(file=io.StringIO(), width=100, no_color=True)
    render(
        ReviewResult(request_ref="r", mode="multi", verified=True,
                     accepted=[_f("stays")], delta=_Delta(),
                     usage={"raw_count": 1, "merged_count": 1, "accepted_count": 1}),
        _request(tmp_path), console,
    )
    out = console.file.getvalue()
    assert "1 fixed since last review" in out
    assert "1 still open from before" in out
