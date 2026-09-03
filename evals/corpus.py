"""Corpus construction.

Takes real source from a real repository, breaks one thing in a known way, and
synthesises the diff that change would have produced. The point of using real
source rather than toy snippets is that the surrounding code has to be plausible
enough for a reviewer to be genuinely uncertain.
"""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from pathlib import Path

from agentgate.filters import is_reviewable_path
from agentgate.schema import FileDiff, Hunk, ReviewRequest

from .mutations import Mutation, all_benign_mutators, all_defect_mutators

# Deliberately no local skip list: the corpus uses the reviewer's own filter, so
# a file can never be seeded with a defect that the reviewer would then refuse to
# look at. Keeping two lists in sync is what produced a corpus made entirely of
# numpy and onnxruntime source, scored as 0% detection.


@dataclass
class Case:
    id: str
    kind: str  # "defect" | "clean"
    request: ReviewRequest
    truth: Mutation | None


def eligible_files(root: Path, min_lines: int = 40, max_lines: int = 400) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if not is_reviewable_path(rel):
            continue
        if p.name.endswith((".test.ts", ".spec.ts", "_test.py")):
            continue
        try:
            n = len(p.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            continue
        if min_lines <= n <= max_lines:
            out.append(p)
    return sorted(out)


def build_request(path: str, original: str, mutated: str, ref: str) -> ReviewRequest:
    """Synthesise the diff the mutation would have produced."""
    a, b = original.splitlines(), mutated.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    hunks: list[Hunk] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        body = [f"-{ln}" for ln in a[i1:i2]] + [f"+{ln}" for ln in b[j1:j2]]
        hunks.append(
            Hunk(
                old_start=i1 + 1,
                old_lines=max(i2 - i1, 1),
                new_start=j1 + 1,
                new_lines=max(j2 - j1, 1),
                content=f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@\n" + "\n".join(body),
            )
        )

    return ReviewRequest(
        source="local",
        ref=ref,
        title=f"eval case: {path}",
        files=[FileDiff(path=path, change_type="modified", hunks=hunks, content_after=mutated)],
    )


def build(
    source_root: Path,
    n_defect: int = 20,
    n_clean: int = 20,
    seed: int = 0,
) -> list[Case]:
    rng = random.Random(seed)
    files = eligible_files(source_root)
    if not files:
        raise RuntimeError(f"No eligible source files under {source_root}")

    defect_mutators = list(all_defect_mutators().items())
    benign_mutators = list(all_benign_mutators().items())
    cases: list[Case] = []

    def attempt(pool, kind: str, want: int) -> None:
        made = 0
        tries = 0
        # Round-robin rather than random choice over mutators. Mutators differ a lot
        # in how often they find a candidate site, so sampling uniformly at random
        # yields a corpus dominated by whichever one fires most easily — which
        # measures that mutator, not the reviewer.
        while made < want and tries < want * 40:
            path = rng.choice(files)
            name, fn = pool[tries % len(pool)]
            tries += 1
            try:
                original = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            lines = original.splitlines()
            mutation = fn(lines, rng)
            if mutation is None:
                continue
            mutated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
            if mutated == original:
                continue
            rel = str(path.relative_to(source_root))
            request = build_request(rel, original, mutated, f"{kind}/{name}/{rel}")
            if not request.reviewable:
                # Belt and braces: a case the reviewer will not look at cannot be
                # scored, and silently keeping it reports 0% detection instead of
                # a broken corpus.
                continue
            cases.append(
                Case(
                    id=f"{kind}-{made:03d}-{name}",
                    kind=kind,
                    request=request,
                    truth=mutation if kind == "defect" else None,
                )
            )
            made += 1

    attempt(defect_mutators, "defect", n_defect)
    attempt(benign_mutators, "clean", n_clean)
    return cases
