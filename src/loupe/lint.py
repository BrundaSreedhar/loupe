"""Run the repository's own tooling before spending a model call.

A linter is instant, free, and constitutionally incapable of inventing a finding.
Anything it catches should never cost a review comment — and more importantly, the
reviewers can be told what it already found so they do not file it again. That is
precision improvement at no cost to recall.

SAFETY: this executes code from the repository being reviewed. An ESLint config is
JavaScript and loading one runs it; the same is true of most project tooling. That
is fine for your own repository and is not fine for a pull request written by
someone you do not trust, so it is opt-in for pull requests and can be disabled
entirely.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TIMEOUT = int(os.getenv("LOUPE_LINT_TIMEOUT", "60"))
MAX_ISSUES = int(os.getenv("LOUPE_LINT_MAX_ISSUES", "60"))


@dataclass(frozen=True)
class LintIssue:
    tool: str
    path: str
    line: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line} [{self.tool} {self.code}] {self.message}"


def _run(cmd: list[str], cwd: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(  # noqa: S603 — running the repo's own
            # tooling is the entire point; see the SAFETY note above.
            cmd, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT, check=False
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("%s failed to run: %s", cmd[0], exc)
        return -1, ""
    return proc.returncode, proc.stdout


def _tool_path(name: str, repo_root: str) -> str | None:
    """Prefer the repo's own pinned copy over whatever is on PATH — a project's
    node_modules version is the one its config was written for."""
    local = Path(repo_root) / "node_modules" / ".bin" / name
    if local.is_file():
        return str(local)
    return shutil.which(name)


def _ruff(paths: list[str], repo_root: str) -> list[LintIssue]:
    exe = _tool_path("ruff", repo_root)
    py = [p for p in paths if p.endswith((".py", ".pyi"))]
    if not exe or not py:
        return []
    _, out = _run([exe, "check", "--output-format=json", "--", *py], repo_root)
    try:
        rows = json.loads(out or "[]")
    except json.JSONDecodeError:
        return []
    return [
        LintIssue(
            tool="ruff",
            path=str(Path(r["filename"]).relative_to(repo_root))
            if str(r["filename"]).startswith(repo_root)
            else r["filename"],
            line=(r.get("location") or {}).get("row", 1),
            code=r.get("code") or "",
            message=r.get("message", ""),
        )
        for r in rows
    ]


def _eslint(paths: list[str], repo_root: str) -> list[LintIssue]:
    exe = _tool_path("eslint", repo_root)
    js = [p for p in paths if p.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"))]
    if not exe or not js:
        return []
    _, out = _run([exe, "--format", "json", "--no-error-on-unmatched-pattern", *js], repo_root)
    try:
        files = json.loads(out or "[]")
    except json.JSONDecodeError:
        return []
    issues: list[LintIssue] = []
    for f in files:
        rel = f.get("filePath", "")
        if rel.startswith(repo_root):
            rel = str(Path(rel).relative_to(repo_root))
        for m in f.get("messages", []):
            issues.append(
                LintIssue(
                    tool="eslint",
                    path=rel,
                    line=m.get("line", 1),
                    code=m.get("ruleId") or "",
                    message=m.get("message", ""),
                )
            )
    return issues


RUNNERS = (_ruff, _eslint)


def available(repo_root: str) -> list[str]:
    """Which linters could run here — for `loupe doctor`, so the answer is knowable
    without executing anything."""
    return [n for n in ("ruff", "eslint") if _tool_path(n, repo_root)]


def run(paths: list[str], repo_root: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for runner in RUNNERS:
        try:
            issues.extend(runner(paths, repo_root))
        except Exception as exc:  # noqa: BLE001 — a broken linter must not stop a review
            log.warning("%s raised: %s", runner.__name__, exc)
    if len(issues) > MAX_ISSUES:
        log.info("%d lint issues; passing the first %d to the reviewers",
                 len(issues), MAX_ISSUES)
        issues = issues[:MAX_ISSUES]
    return issues


def as_prompt_section(issues: list[LintIssue]) -> str:
    """What the reviewers are told has already been covered."""
    if not issues:
        return ""
    lines = "\n".join(f"  {i.render()}" for i in issues)
    return (
        "ALREADY REPORTED BY THIS PROJECT'S OWN TOOLING\n"
        "These were found by the repository's linters before you were called. They "
        "are already on their way to the author.\n\n"
        f"{lines}\n\n"
        "Do not report any of these, and do not report the same defect in different "
        "words. Anything a linter can catch is not worth your attention — spend it "
        "on what a linter cannot see.\n"
    )
