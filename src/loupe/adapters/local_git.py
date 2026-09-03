"""Local git adapter: review a ref range against the working tree."""

from __future__ import annotations

import difflib
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
    """Tree-ish, not commit. `git diff` accepts any tree-ish, and the empty-tree
    hash used to review an initial commit is a tree with no commit behind it —
    checking for `^{commit}` would reject the very ref we suggest."""
    try:
        _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{tree}}"], cwd)
    except GitError:
        return False
    return True


def _commit_count(cwd: str) -> int:
    try:
        return int(_git(["rev-list", "--count", "HEAD"], cwd).strip())
    except (GitError, ValueError):
        return 0


def _empty_tree(cwd: str) -> str:
    """The hash of git's empty tree — diffing against it yields the whole of the
    first commit, which is the only way to review a repository's initial import."""
    return _git(["hash-object", "-t", "tree", "/dev/null"], cwd).strip()


def _untracked(root: str) -> list[str]:
    """Files Git deliberately omits from every normal diff.

    A local reviewer should see a new source file before the author remembers to
    stage it. ``--exclude-standard`` honours the repository's own ignore rules.
    """
    return [p for p in _git(["ls-files", "--others", "--exclude-standard"], root).splitlines() if p]


def _untracked_diff(root: str) -> list:
    from .diffparse import parse_unified_diff

    files = []
    for path in _untracked(root):
        disk = Path(root) / path
        if not disk.is_file():
            continue
        try:
            content = disk.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        body = "".join(
            difflib.unified_diff(
                [], content.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{path}"
            )
        )
        parsed = parse_unified_diff(
            f"diff --git a/{path} b/{path}\nnew file mode 100644\n{body}"
        )
        for fd in parsed:
            fd.content_after = content
        files.extend(parsed)
    return files


def _suggest(ref: str, root: str) -> str:
    """A ref failing on a young repository is nearly always the ~N walking off the
    end of history, which the bare git error does not say."""
    n = _commit_count(root)
    if not (ref.startswith("HEAD~") and n):
        return ""
    return (
        f"\nThis repository has only {n} commit{'s' if n != 1 else ''}, "
        f"so there is no {ref} to diff against.\n\n"
        "Try one of:\n"
        "  loupe local HEAD                       review uncommitted changes\n"
        "  loupe local --staged                   review what is staged\n"
        "  loupe local HEAD~1 --repo-root ~/repo  review a repo with history\n"
        f"  loupe local {_empty_tree(root)}\n"
        "                                          review the first commit whole"
    )


def load(
    ref: str = "HEAD~1", repo_root: str = ".", staged: bool = False, include_untracked: bool = True
) -> ReviewRequest:
    root = str(Path(repo_root).resolve())
    has_head = _resolves("HEAD", root)

    if not staged and not _resolves(ref, root):
        if not has_head:
            raise GitError(
                f"{root} has no commits yet, so there is nothing to diff against. "
                "Stage some changes and use `--staged`, or make a commit first."
            )
        raise GitError(f"Cannot resolve ref {ref!r} in {root}.{_suggest(ref, root)}")
    if staged and not has_head:
        raise GitError(
            f"{root} has no commits yet, so there is no HEAD for the index to be "
            "compared against. Make an initial commit first."
        )

    args = ["diff", "--no-color", "--find-renames"]
    args += ["--cached"] if staged else [ref]
    diff_text = _git(args, root)

    files = parse_unified_diff(diff_text)
    if include_untracked:
        files.extend(_untracked_diff(root))
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
