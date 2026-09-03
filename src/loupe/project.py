"""Per-project configuration in a `.loupe/` folder.

A repository carries its own review settings the way it carries its own linter
config, so a checkout is self-describing and settings travel with the code:

    .loupe/
      config.env    settings — same keys as .env
      rules.md      review guidance specific to this codebase
      ignore        extra paths to skip, one glob per line

`rules.md` is the part worth having. Generic reviewers find generic bugs; the
defects a codebase produces repeatedly are the ones only its own team can name.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from pathlib import Path

log = logging.getLogger(__name__)

DIR_NAME = ".loupe"
MAX_RULES_CHARS = 8000


def find_dir(start: Path | str) -> Path | None:
    """Nearest `.loupe/` at or above `start`."""
    here = Path(start).resolve()
    for candidate in (here, *here.parents):
        found = candidate / DIR_NAME
        if found.is_dir():
            return found
    return None


def load_rules(repo_root: Path | str) -> str | None:
    """Project-specific review guidance, if the repo defines any."""
    directory = find_dir(repo_root)
    if directory is None:
        return None
    path = directory / "rules.md"
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    if not text:
        return None
    if len(text) > MAX_RULES_CHARS:
        log.warning("%s is %d chars; using the first %d", path, len(text), MAX_RULES_CHARS)
        text = text[:MAX_RULES_CHARS]
    log.info("using project rules from %s", path)
    return text


def load_ignore(repo_root: Path | str) -> list[str]:
    """Extra path globs to skip, beyond the built-in filters."""
    directory = find_dir(repo_root)
    if directory is None:
        return []
    path = directory / "ignore"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def is_ignored(path: str, patterns: list[str]) -> bool:
    return any(fnmatch(path, p) or fnmatch(path, f"*/{p}") for p in patterns)


RULES_TEMPLATE = """\
# Review rules for this codebase

Written by the maintainers, and given to every reviewer alongside the diff. Use it
for the defects this codebase produces repeatedly — the ones a generic reviewer
has no way of knowing about.

Be specific about what is wrong and why. Vague guidance produces vague findings.

## Examples — replace these

- All database access goes through `db/gateway.ts`. A direct `pg.query` call is a
  high-severity finding even when the SQL itself is safe.
- Money is always integer cents. A float touching a currency value is a bug.
- Anything under `handlers/` runs untrusted input. Validation belongs at the top
  of the handler, not in the helpers it calls.
"""

IGNORE_TEMPLATE = """\
# Extra paths to skip, one glob per line, beyond the built-in filters.
# Lockfiles, docs, vendored and minified files are already skipped.

# src/generated/*
# *.pb.ts
"""

CONFIG_TEMPLATE = """\
# Settings for this repository only. Same keys as the global config, and read
# before it — but a variable set in your shell still wins.

# LOUPE_MAX_REPORTED=8
# LOUPE_REVIEW_TOKEN_BUDGET=120000
"""


def scaffold(repo_root: Path | str, force: bool = False) -> tuple[Path, list[str], list[str]]:
    """Create `.loupe/` in a repository. Returns (dir, written, skipped)."""
    directory = Path(repo_root).resolve() / DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for name, body in (
        ("rules.md", RULES_TEMPLATE),
        ("ignore", IGNORE_TEMPLATE),
        ("config.env", CONFIG_TEMPLATE),
    ):
        target = directory / name
        if target.exists() and not force:
            skipped.append(name)
            continue
        target.write_text(body)
        written.append(name)
    return directory, written, skipped
