"""What is worth sending to a reviewer at all.

The cheapest quality win available: a security specialist asked to review a
lockfile or a README will find something to say, and all of it is noise. Filtering
by path costs nothing and removes a whole class of false positive before any model
call is made.
"""

from __future__ import annotations

from pathlib import PurePosixPath

SOURCE_SUFFIXES = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".rb", ".php", ".cs", ".swift",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".sh", ".bash", ".sql",
}

# Directories whose contents are not authored by this change.
SKIP_DIRS = {
    "node_modules", "vendor", "dist", "build", "out", "target",
    ".next", ".nuxt", "__pycache__", ".venv", "venv", "coverage",
    "migrations", ".git", "site-packages",
}

SKIP_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "uv.lock", "Cargo.lock", "composer.lock", "Gemfile.lock",
}

# Substrings that mark a file as generated or not hand-written.
SKIP_MARKERS = (".min.", ".bundle.", ".generated.", "_pb2.", ".pb.go", ".d.ts")


def _is_env_dir(part: str) -> bool:
    """Virtualenvs get named all sorts of things — .venv, venv, .venv-tts, env311.
    An exact-name list misses every variant, which is how third-party source ends
    up being reviewed as if it were yours."""
    low = part.lower()
    return low.startswith((".venv", "venv", ".env", "env-")) or low in {"env", ".tox"}


def is_reviewable_path(path: str) -> bool:
    p = PurePosixPath(path)
    parts = set(p.parts)
    if SKIP_DIRS & parts:
        return False
    if any(_is_env_dir(part) for part in p.parts):
        return False
    if p.name in SKIP_NAMES:
        return False
    if any(marker in p.name for marker in SKIP_MARKERS):
        return False
    return p.suffix.lower() in SOURCE_SUFFIXES


def is_test_path(path: str) -> bool:
    """Tests are reviewed, but the maintainability reviewer should not complain
    that a test file has no tests."""
    name = PurePosixPath(path).name
    return (
        name.startswith("test_")
        or "_test." in name
        or ".test." in name
        or ".spec." in name
        or "tests" in PurePosixPath(path).parts
    )
