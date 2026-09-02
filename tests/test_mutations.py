import random

import pytest

from evals.corpus import build_request
from evals.mutations import all_benign_mutators, all_defect_mutators

SAMPLE = '''\
async def handler(request, db):
    if not request.user_id:
        return None
    total = high - low
    for i in range(len(items)):
        process(items[i])
    rows = await db.fetch("SELECT * FROM users WHERE id = %s", request.user_id)
    if rows == []:
        return None
    try:
        return rows[0]
    except KeyError as exc:
        raise exc
'''


@pytest.mark.parametrize("name", sorted(all_defect_mutators()))
def test_defect_mutator_changes_source_and_records_its_line(name):
    fn = all_defect_mutators()[name]
    lines = SAMPLE.splitlines()
    mutation = fn(lines, random.Random(0))
    if mutation is None:
        pytest.skip(f"{name} found no candidate site in the sample")
    assert "\n".join(lines) != SAMPLE
    assert 1 <= mutation.line <= len(lines)


@pytest.mark.parametrize("name", sorted(all_benign_mutators()))
def test_benign_mutator_reports_no_defect_category(name):
    fn = all_benign_mutators()[name]
    lines = SAMPLE.splitlines()
    mutation = fn(lines, random.Random(0))
    if mutation is None:
        pytest.skip(f"{name} found no candidate site in the sample")
    assert mutation.category == "none"


def test_synthesised_diff_anchors_the_mutated_line():
    """The ground-truth line must actually appear as changed in the diff the
    reviewer is shown, or recall is being measured against a line nobody saw."""
    lines = SAMPLE.splitlines()
    mutation = all_defect_mutators()["flip_equality"](lines, random.Random(0))
    assert mutation is not None
    mutated = "\n".join(lines)
    request = build_request("sample.py", SAMPLE, mutated, "test")
    assert mutation.line in request.files[0].changed_lines
