"""A frozen corpus: the same bugs, every run, forever.

The generated corpus is a function of a source repository and a seed, which makes
it reproducible only while that repository stays exactly as it was. Add a file,
land a refactor, or point `--source` somewhere else and the cases change
underneath you — so a number from last month cannot be compared with one from
today, which removes most of the reason to measure anything.

Freezing solves that without giving up real code. The mutators run once against a
real repository, and the resulting cases are written out as JSON and committed.
From then on every run scores the identical twenty bugs, and a change in the
number means a change in the reviewer.

One file per case on purpose: a bad case can be deleted, a case shows up as a
readable diff when it is added, and `git log` on this directory says how the
measurement has changed over time.

**What a frozen case does and does not carry.** The mutated file travels with the
case, so a single-file bug is entirely self-contained. `repo_root` is recorded as
provenance but the repository itself is not vendored, so a case whose defect needs
repository context — the cross-file ones — will only exercise that context on a
machine where that repo is still present at that path. `load` says so rather than
letting it degrade silently.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from loupe.schema import ReviewRequest

from .corpus import Case
from .mutations import Mutation

# Bumped when the on-disk shape changes. A loader that silently accepts an older
# shape produces cases that are subtly not what was frozen.
SCHEMA = 1

MANIFEST = "manifest.json"


def _source_commit(root: Path) -> str:
    """The commit the cases were cut from, when the source is a git checkout.

    Provenance rather than machinery: the cases carry their own source, but "which
    revision of which repo did these bugs come from" is the first question anyone
    asks about a corpus, and it is unanswerable later if it is not recorded now.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _to_dict(case: Case) -> dict:
    return {
        "schema": SCHEMA,
        "id": case.id,
        "kind": case.kind,
        "truth": asdict(case.truth) if case.truth else None,
        "request": case.request.model_dump(mode="json"),
    }


def _from_dict(data: dict, where: Path) -> Case:
    got = data.get("schema")
    if got != SCHEMA:
        raise ValueError(
            f"{where.name} was written by schema {got}, this loader is {SCHEMA}. "
            "Re-freeze the seed set rather than scoring against a shape this code "
            "does not understand."
        )
    truth = data.get("truth")
    return Case(
        id=data["id"],
        kind=data["kind"],
        request=ReviewRequest.model_validate(data["request"]),
        truth=Mutation(**truth) if truth else None,
    )


def freeze(cases: list[Case], directory: Path, source: Path) -> list[Path]:
    """Write cases out as committable JSON. Returns the files written."""
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("case-*.json"):
        # A freeze replaces the set. Leaving old files behind would silently mix
        # two generations of corpus in every later run.
        stale.unlink()

    written: list[Path] = []
    for case in sorted(cases, key=lambda c: c.id):
        path = directory / f"case-{case.id}.json"
        path.write_text(json.dumps(_to_dict(case), indent=2, sort_keys=True) + "\n")
        written.append(path)

    (directory / MANIFEST).write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "frozen_at": datetime.now(UTC).isoformat(),
                "source": str(source),
                "source_commit": _source_commit(source),
                "n_defect": sum(1 for c in cases if c.kind == "defect"),
                "n_clean": sum(1 for c in cases if c.kind == "clean"),
                "mutators": describe(cases),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return written


def load(directory: Path) -> tuple[list[Case], dict]:
    """Every frozen case, in a stable order, with the manifest beside it."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No seed set at {directory}. Create one with:\n"
            f"  python -m evals.run_eval freeze --source <repo>"
        )
    files = sorted(directory.glob("case-*.json"))
    if not files:
        raise FileNotFoundError(f"{directory} holds no case-*.json files.")

    cases = [_from_dict(json.loads(p.read_text()), p) for p in files]
    manifest_path = directory / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    return cases, manifest


def describe(cases: list[Case]) -> dict[str, int]:
    """How many cases per mutator — the check on whether the set is balanced."""
    out: dict[str, int] = {}
    for c in cases:
        key = c.truth.name if c.truth else "clean"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def needs_source_repo(cases: list[Case]) -> list[str]:
    """Case ids whose defect is only visible with the source repository present.

    Cross-file cases carry the calling file but not the file holding the callee, so
    on a machine without that checkout they measure the reviewer with the context
    removed — which looks like a reviewer that missed them.
    """
    return [c.id for c in cases if c.id.startswith("defect-x")]
