"""Unified-diff parser shared by both adapters.

Written by hand rather than pulled in as a dependency because the only thing this
system genuinely needs from a diff is accurate post-change line numbers, and that
is exactly the part wrapper libraries tend to get subtly wrong on renames and
no-newline-at-EOF.
"""

from __future__ import annotations

import re

from ..schema import FileDiff, Hunk

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk_lines: list[str] = []
    hunk_header: tuple[int, int, int, int] | None = None

    def flush_hunk() -> None:
        nonlocal hunk_lines, hunk_header
        if current is not None and hunk_header is not None:
            os_, ol, ns, nl = hunk_header
            current.hunks.append(
                Hunk(
                    old_start=os_,
                    old_lines=ol,
                    new_start=ns,
                    new_lines=nl,
                    content="\n".join(hunk_lines),
                )
            )
        hunk_lines = []
        hunk_header = None

    for line in text.splitlines():
        if line.startswith("diff --git "):
            flush_hunk()
            if current is not None:
                files.append(current)
            # `diff --git a/x b/y` — take the b-side as the path.
            parts = line.split(" b/", 1)
            path = parts[1] if len(parts) == 2 else line.split()[-1]
            current = FileDiff(path=path)
            continue

        if current is None:
            continue

        if line.startswith("new file mode"):
            current.change_type = "added"
        elif line.startswith("deleted file mode"):
            current.change_type = "deleted"
        elif line.startswith("rename from "):
            current.old_path = line[len("rename from ") :].strip()
            current.change_type = "renamed"
        elif line.startswith(("GIT binary patch", "Binary files ")):
            current.is_binary = True
        elif line.startswith("--- ") and line[4:].strip() not in ("/dev/null",):
            current.old_path = current.old_path or line[4:].strip().removeprefix("a/")
        elif (m := _HUNK.match(line)) is not None:
            flush_hunk()
            hunk_header = (
                int(m.group(1)),
                int(m.group(2) or 1),
                int(m.group(3)),
                int(m.group(4) or 1),
            )
            hunk_lines = [line]
        elif hunk_header is not None:
            hunk_lines.append(line)

    flush_hunk()
    if current is not None:
        files.append(current)
    return files


def added_line_numbers(hunk: Hunk) -> list[int]:
    """Post-change line numbers of `+` lines only — what a finding may anchor to."""
    out: list[int] = []
    n = hunk.new_start
    for raw in hunk.content.splitlines()[1:]:  # skip the @@ header
        if raw.startswith("+"):
            out.append(n)
            n += 1
        elif raw.startswith("-"):
            continue
        else:
            n += 1
    return out
