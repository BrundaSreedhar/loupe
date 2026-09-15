"""The frozen corpus.

Everything here guards one property: a case that loads must be the case that was
frozen. If the round trip loses the mutated source, or the ground-truth line, or
the changed-line set the diff produces, then the seed set scores a reviewer against
something other than what was committed — and no column in the report would show it.
"""

from __future__ import annotations

import json

import pytest

from evals.corpus import Case, build_request
from evals.mutations import Mutation
from evals.seedset import MANIFEST, SCHEMA, describe, freeze, load, needs_source_repo

SOURCE = "def charge(account, amount):\n    return account - amount\n"
MUTATED = "def charge(account, amount):\n    return amount - account\n"


def _case(case_id: str = "defect-000-swap_operands", kind: str = "defect") -> Case:
    truth = (
        Mutation(
            name="swap_operands",
            category="correctness",
            line=2,
            before="return account - amount",
            after="return amount - account",
            description="Operands swapped, negating the result.",
        )
        if kind == "defect"
        else None
    )
    return Case(
        id=case_id,
        kind=kind,
        request=build_request("pkg/charge.py", SOURCE, MUTATED, "test", repo_root="/src"),
        truth=truth,
    )


def test_a_frozen_case_loads_back_identical(tmp_path):
    original = _case()
    freeze([original], tmp_path, tmp_path)
    (loaded,), _ = load(tmp_path)

    assert loaded.id == original.id
    assert loaded.kind == original.kind
    assert loaded.truth == original.truth
    assert loaded.request.files[0].content_after == MUTATED


def test_the_diff_survives_the_round_trip(tmp_path):
    """The hunks are what the reviewer is shown and what `changed_lines` is derived
    from. A case that loads with different hunks is scored against a diff nobody
    ever froze."""
    original = _case()
    freeze([original], tmp_path, tmp_path)
    (loaded,), _ = load(tmp_path)

    assert loaded.request.files[0].changed_lines == original.request.files[0].changed_lines
    assert loaded.truth.line in loaded.request.files[0].changed_lines


def test_a_clean_control_keeps_its_absent_ground_truth(tmp_path):
    freeze([_case("clean-000-add_comment", kind="clean")], tmp_path, tmp_path)
    (loaded,), _ = load(tmp_path)
    assert loaded.kind == "clean"
    assert loaded.truth is None


def test_cases_load_in_a_stable_order(tmp_path):
    ids = ["defect-002-a", "defect-000-b", "defect-001-c"]
    freeze([_case(i) for i in ids], tmp_path, tmp_path)
    first, _ = load(tmp_path)
    second, _ = load(tmp_path)
    assert [c.id for c in first] == [c.id for c in second] == sorted(ids)


def test_the_manifest_records_where_the_cases_came_from(tmp_path):
    """The first question anyone asks about a corpus, and unanswerable later if it
    is not written down now."""
    freeze([_case()], tmp_path, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    assert manifest["schema"] == SCHEMA
    assert manifest["source"] == str(tmp_path)
    assert manifest["n_defect"] == 1
    assert manifest["frozen_at"]
    assert manifest["mutators"] == {"swap_operands": 1}


def test_refreezing_removes_the_previous_generation(tmp_path):
    """Two generations of corpus in one directory is the worst outcome available:
    every run silently scores a mixture."""
    freeze([_case("defect-000-old"), _case("defect-001-old")], tmp_path, tmp_path)
    freeze([_case("defect-000-new")], tmp_path, tmp_path)
    loaded, _ = load(tmp_path)
    assert [c.id for c in loaded] == ["defect-000-new"]


def test_an_unknown_schema_is_refused_rather_than_guessed(tmp_path):
    freeze([_case()], tmp_path, tmp_path)
    path = next(tmp_path.glob("case-*.json"))
    data = json.loads(path.read_text())
    data["schema"] = SCHEMA + 99
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Re-freeze"):
        load(tmp_path)


def test_a_missing_seed_set_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="freeze"):
        load(tmp_path / "nothing-here")


def test_an_empty_directory_is_not_an_empty_corpus(tmp_path):
    """Zero cases would score as a confident 0% detection. Same rule as the rest of
    the harness: never report a failure as a measurement."""
    (tmp_path / "seed").mkdir()
    with pytest.raises(FileNotFoundError, match="no case"):
        load(tmp_path / "seed")


def test_cross_file_cases_are_flagged_as_needing_the_source_repo():
    """They carry the calling file but not the callee's, so without the checkout
    they measure the reviewer with its context removed."""
    cases = [_case("defect-x000-swapped_call_arguments"), _case("defect-000-swap_operands")]
    assert needs_source_repo(cases) == ["defect-x000-swapped_call_arguments"]


def test_describe_counts_clean_controls_separately():
    assert describe([_case(), _case("clean-000-x", kind="clean")]) == {
        "clean": 1,
        "swap_operands": 1,
    }


# ─── the committed set itself ────────────────────────────────────────────────


def test_the_committed_seed_set_is_loadable_and_balanced():
    """Guards the artefact, not just the machinery. A committed set that fails to
    load, or that one mutator dominates, is a broken measurement sitting in the
    repository looking authoritative."""
    from pathlib import Path

    directory = Path(__file__).resolve().parents[1] / "evals" / "seed"
    if not directory.is_dir():
        pytest.skip("no seed set committed")

    cases, manifest = load(directory)
    defects = [c for c in cases if c.kind == "defect"]
    cleans = [c for c in cases if c.kind == "clean"]

    assert len(defects) >= 20, "fewer than 20 seeded defects"
    assert cleans, "detection without clean controls is not a measurement"

    counts = {k: v for k, v in describe(cases).items() if k != "clean"}
    assert len(counts) >= 8, f"only {len(counts)} defect shapes: {counts}"
    assert max(counts.values()) <= len(defects) // 4, f"one mutator dominates: {counts}"

    for c in defects:
        assert c.truth is not None
        assert c.truth.line in c.request.files[0].changed_lines, (
            f"{c.id}: ground-truth line is not in the diff the reviewer sees"
        )
    assert manifest.get("source_commit"), "no provenance recorded"
