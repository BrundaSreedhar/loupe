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


# ─── windows snap to whole definitions ──────────────────────────────────────


def _module(filler: int, body: int) -> tuple[str, int, int]:
    """A module of small filler functions, then one long `wide`. Returns the
    source and the line range `wide` occupies, so the tests assert against real
    positions instead of hand-counted ones."""
    parts = [f"def filler_{i}(a):\n    return a + {i}\n" for i in range(filler)]
    steps = "\n".join(f"    step_{j} = {j}" for j in range(1, body + 1))
    parts.append(f"def wide(x):\n{steps}\n    return x\n")
    parts.append("def narrow(y):\n    return y\n")
    content = "\n\n".join(parts) + "\n"
    start = next(
        i for i, line in enumerate(content.splitlines(), 1) if line.startswith("def wide")
    )
    return content, start, start + body + 1


def _diff(content: str, new_start: int, new_end: int, path: str = "pkg/mod.py"):
    from loupe.schema import FileDiff, Hunk

    return FileDiff(
        path=path,
        change_type="modified",
        content_after=content,
        hunks=[
            Hunk(
                old_start=new_start,
                old_lines=new_end - new_start + 1,
                new_start=new_start,
                new_lines=new_end - new_start + 1,
                content="",
            )
        ],
    )


def test_a_windowed_file_shows_the_whole_enclosing_function(monkeypatch):
    """Half a function is the worst thing to hand a reviewer: no guard clause above
    the change, no return below it. Padding guesses where the function starts; the
    parser knows.

    The budget sits between the whole file (~1000 tokens) and `wide` alone (~280),
    so the file must be windowed but the function still fits whole.
    """
    from loupe import context

    content, start, end = _module(filler=40, body=40)
    monkeypatch.setattr(context, "FILE_TOKEN_BUDGET", 500)
    monkeypatch.setattr(context, "WINDOW_PADDING", 5)

    ctx = context.build_file_context(_diff(content, end - 2, end - 2))

    assert ctx.strategy == "windowed"
    assert "def wide(x):" in ctx.content
    assert f" {start}|" in ctx.content
    assert ctx.definitions == ["wide"]


def test_padding_alone_would_have_missed_it(monkeypatch):
    """The control for the test above. Without the snap the signature is 40 lines
    above the window and the reviewer never sees it."""
    from loupe import context

    content, _, end = _module(filler=40, body=40)
    monkeypatch.setattr(context, "WINDOW_PADDING", 5)

    total = len(content.splitlines())
    fd = _diff(content, end - 2, end - 2)
    unsnapped = context._render_spans(
        content, {end - 2}, context._spans(fd, total, []), total
    )
    assert "def wide(x):" not in unsnapped


def test_an_enclosing_definition_too_big_to_show_falls_back_to_padding(monkeypatch):
    """Snapping to a 400-line function would spend the whole file budget on one
    hunk. Over budget, the padded window at least keeps the change in view."""
    from loupe import context

    content, _, end = _module(filler=40, body=400)
    monkeypatch.setattr(context, "FILE_TOKEN_BUDGET", 500)
    monkeypatch.setattr(context, "WINDOW_PADDING", 3)

    changed_line = end - 2
    ctx = context.build_file_context(_diff(content, changed_line, changed_line))

    assert ctx.strategy == "windowed"
    assert "def wide(x):" not in ctx.content
    # ...but the change itself is still there, which is the point of the fallback.
    assert content.splitlines()[changed_line - 1].strip() in ctx.content


def test_a_whole_file_context_still_records_what_was_touched():
    """The names are recorded whether or not the file needed windowing — the
    progress line counts them either way."""
    from loupe import context

    content, _, end = _module(filler=1, body=2)
    ctx = context.build_file_context(_diff(content, end - 1, end - 1))
    assert ctx.strategy == "whole_file"
    assert ctx.definitions == ["wide"]


def test_a_non_python_file_gets_no_definitions():
    """Same rule as expansion: a language without a parser here contributes
    nothing, rather than a regex's guess at where a function starts."""
    from loupe import context

    fd = _diff("function wide(x) {\n  return x;\n}\n", 2, 2, path="pkg/mod.ts")
    assert context.build_file_context(fd).definitions == []
