"""Local git adapter: review a ref range against the working tree."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..schema import ReviewRequest
from .diffparse import parse_unified_diff


class GitError(RuntimeError):
    """A git invocation failed — usually an unresolvable ref."""


def _git(args: list[str], cwd: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _resolves(ref: str, cwd: str) -> bool:
    try:
        _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd)
    except GitError:
        return False
    return True


def load(ref: str = "HEAD~1", repo_root: str = ".", staged: bool = False) -> ReviewRequest:
    root = str(Path(repo_root).resolve())
    has_head = _resolves("HEAD", root)

    if not staged and not _resolves(ref, root):
        if not has_head:
            raise GitError(
                f"{root} has no commits yet, so there is nothing to diff against. "
                "Stage some changes and use `--staged`, or make a commit first."
            )
        raise GitError(f"Cannot resolve ref {ref!r} in {root}.")
    if staged and not has_head:
        raise GitError(
            f"{root} has no commits yet, so there is no HEAD for the index to be "
            "compared against. Make an initial commit first."
        )

    args = ["diff", "--no-color", "--find-renames"]
    args += ["--cached"] if staged else [ref]
    diff_text = _git(args, root)

    files = parse_unified_diff(diff_text)
    for f in files:
        if f.is_binary or f.change_type == "deleted":
            continue
        # Working tree is the post-change state for a local review.
        p = Path(root) / f.path
        try:
            f.content_after = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            f.is_binary = True

    head = _git(["rev-parse", "--short", "HEAD"], root).strip()
    return ReviewRequest(
        source="local",
        ref="staged" if staged else ref,
        title=f"Local diff {'--cached' if staged else ref} @ {head}",
        files=files,
        repo_root=root,
    )
