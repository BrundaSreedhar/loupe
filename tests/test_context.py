from loupe.context import render_numbered
from loupe.nodes._common import drop_ungrounded, valid_lines
from loupe.schema import FileContext, Finding

SRC = "\n".join(f"line {i}" for i in range(1, 11))


def _ctx(content: str) -> FileContext:
    return FileContext(path="a.py", content=content, strategy="whole_file", tokens=10)


def test_changed_lines_are_marked():
    out = render_numbered(SRC, changed={3, 4})
    rows = out.splitlines()
    assert rows[2].startswith(">")
    assert rows[3].startswith(">")
    assert rows[4].startswith(" ")


def test_windowing_respects_bounds():
    out = render_numbered(SRC, changed=set(), lo=4, hi=6)
    assert len(out.splitlines()) == 3
    assert "line 4" in out and "line 6" in out and "line 7" not in out


def test_valid_lines_recovers_numbers():
    assert valid_lines(_ctx(render_numbered(SRC, changed={2}))) == set(range(1, 11))


def _finding(**kw) -> Finding:
    base = {
        "id": "x-1",
        "produced_by": "correctness",
        "file": "a.py",
        "line": 3,
        "category": "correctness",
        "severity": "high",
        "summary": "s",
        "failure_scenario": "f",
        "confidence": 0.9,
    }
    return Finding(**{**base, **kw})


def test_drops_findings_on_unseen_files():
    contexts = {"a.py": _ctx(render_numbered(SRC, changed={3}))}
    kept, dropped = drop_ungrounded([_finding(file="other.py")], contexts)
    assert kept == [] and len(dropped) == 1


def test_drops_findings_on_lines_outside_context():
    contexts = {"a.py": _ctx(render_numbered(SRC, changed={3}, lo=1, hi=5))}
    kept, _ = drop_ungrounded([_finding(line=9)], contexts)
    assert kept == []


def test_keeps_grounded_findings():
    contexts = {"a.py": _ctx(render_numbered(SRC, changed={3}))}
    kept, dropped = drop_ungrounded([_finding(line=3)], contexts)
    assert len(kept) == 1 and dropped == []
