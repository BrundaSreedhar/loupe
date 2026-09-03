"""Per-project configuration in a `.agentgate/` folder.

A repository carries its own review settings the way it carries its own linter
config, so a checkout is self-describing and settings travel with the code:

    .agentgate/
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

DIR_NAME = ".agentgate"
MAX_RULES_CHARS = 8000


def find_dir(start: Path | str) -> Path | None:
    """Nearest `.agentgate/` at or above `start`."""
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
