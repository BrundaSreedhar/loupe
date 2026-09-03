"""Remembering what was already reported.

A reviewer that repeats the same four comments after every push gets muted by
Thursday. Once findings have a stable identity across runs, the review can report
what changed instead of reprinting the list.

State lives outside the repository. Writing it into the working tree would put a
machine-generated file into the diff — which the reviewer would then be asked to
review.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .schema import Finding

log = logging.getLogger(__name__)

SCHEMA = 1


def state_dir() -> Path:
    base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return Path(os.getenv("LOUPE_STATE_DIR", base / "loupe" / "state"))


def _slug(repo: str, branch: str) -> str:
    """One file per repository and branch. Hashed because a repo identity can be a
    filesystem path, a URL, or an owner/name pair."""
    digest = hashlib.sha256(f"{repo}\x00{branch}".encode()).hexdigest()[:16]
    tail = "".join(c if c.isalnum() else "-" for c in Path(repo).name)[:32]
    return f"{tail}-{branch.replace('/', '-')[:24]}-{digest}.json"


@dataclass
class Delta:
    """What changed since the last review of this branch."""

    new: list[Finding]
    persisting: list[Finding]
    resolved: list[str]        # fingerprints, the findings themselves are gone
    first_run: bool

    @property
    def worth_reporting(self) -> list[Finding]:
        return self.new


def path_for(repo: str, branch: str) -> Path:
    return state_dir() / _slug(repo, branch)


def load(repo: str, branch: str) -> dict[str, dict]:
    p = path_for(repo, branch)
    if not p.is_file():
        return {}
    try:
        blob = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read review memory at %s (%s); treating as empty", p, exc)
        return {}
    if blob.get("schema") != SCHEMA:
        log.info("review memory at %s is an older format; starting fresh", p)
        return {}
    return blob.get("findings") or {}


def save(repo: str, branch: str, findings: list[Finding]) -> None:
    p = path_for(repo, branch)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "schema": SCHEMA,
            "repo": repo,
            "branch": branch,
            "updated": time.time(),
            "findings": {
                f.fingerprint: {
                    "file": f.file,
                    "line": f.line,
                    "category": f.category,
                    "severity": f.severity,
                    "summary": f.summary,
                }
                for f in findings
                if f.fingerprint
            },
        }, indent=2))
    except OSError as exc:
        # Losing memory costs a repeated comment next time, not a wrong review.
        log.warning("could not write review memory to %s: %s", p, exc)


def diff(previous: dict[str, dict], current: list[Finding], first_run: bool) -> Delta:
    seen = {f.fingerprint for f in current if f.fingerprint}
    return Delta(
        new=[f for f in current if f.fingerprint and f.fingerprint not in previous],
        persisting=[f for f in current if f.fingerprint and f.fingerprint in previous],
        resolved=[fp for fp in previous if fp not in seen],
        first_run=first_run,
    )
