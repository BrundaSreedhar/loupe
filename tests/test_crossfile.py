"""Cross-file corpus cases.

These exist to measure one thing: whether showing a reviewer the definitions its
change calls makes it catch defects it could not otherwise see. That only works
if the seeded defect really is invisible in the changed file — so most of what is
tested here is which calls the builder refuses to touch.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

from evals.corpus import crossfile_cases, eligible_files
from evals.crossfile import drop_last_argument, find_sites, swap_first_arguments
from loupe.index import build as build_index

CHARGES = '''\
def charge(user, amount):
    """Take money from a user."""
    return user.balance - amount


def refund(user):
    return user.balance
'''

CALLER = '''\
from billing.charges import charge


def checkout(user, total):
    receipt = charge(user, total)
    return receipt
'''


def make_repo(root: Path, caller: str = CALLER) -> None:
    (root / "billing").mkdir(parents=True)
    (root / "billing" / "__init__.py").write_text("")
    (root / "billing" / "charges.py").write_text(CHARGES)
    (root / "billing" / "checkout.py").write_text(caller)


def sites_for(root: Path, caller: str = CALLER):
    make_repo(root, caller)
    index = build_index(root, max_files=100)
    return find_sites(root, root / "billing" / "checkout.py", index)


def test_a_call_across_files_is_a_site(tmp_path):
    sites = sites_for(tmp_path)
    assert len(sites) == 1
    assert sites[0].callee.path == "billing/charges.py"
    assert sites[0].params == ["user", "amount"]
    assert sites[0].line == 5


def test_dropping_an_argument_leaves_code_that_still_parses(tmp_path):
    site = sites_for(tmp_path)[0]
    mutated, mutation = drop_last_argument(site, random.Random(0))

    ast.parse(mutated)  # a corpus case that will not parse measures nothing
    assert "charge(user)" in mutated
    assert mutation.line == site.line
    assert "billing/charges.py" in mutation.description


def test_swapping_arguments_keeps_the_call_valid(tmp_path):
    site = sites_for(tmp_path)[0]
    mutated, mutation = swap_first_arguments(site, random.Random(0))

    ast.parse(mutated)
    assert "charge(total, user)" in mutated
    assert mutation.line == site.line


def test_the_defect_is_not_visible_in_the_changed_file(tmp_path):
    """The whole premise. If the caller still contains the definition, a reviewer
    reading one file could catch it and the case measures nothing."""
    site = sites_for(tmp_path)[0]
    mutated, _ = drop_last_argument(site, random.Random(0))

    assert "def charge" not in mutated


def test_a_call_to_a_function_in_the_same_file_is_not_a_site(tmp_path):
    caller = "def helper(a, b):\n    return a + b\n\n\ndef run(x, y):\n    return helper(x, y)\n"
    assert sites_for(tmp_path, caller) == []


def test_a_call_into_a_library_is_not_a_site(tmp_path):
    """The callee's signature has to be in this repository, or the reviewer could
    not have found it either and the case is unfair rather than hard."""
    caller = "from json import dumps\n\n\ndef run(x, y):\n    return dumps(x, y)\n"
    assert sites_for(tmp_path, caller) == []


def test_calls_with_keywords_or_defaults_are_left_alone(tmp_path):
    """Dropping an argument that has a default is not a defect."""
    caller = (
        "from billing.charges import charge\n\n\n"
        "def checkout(user, total):\n    return charge(user, amount=total)\n"
    )
    assert sites_for(tmp_path, caller) == []


def test_a_single_argument_call_is_left_alone(tmp_path):
    """Dropping the only argument leaves `refund()`, which reads as obviously
    wrong without knowing anything about `refund`."""
    caller = (
        "from billing.charges import refund\n\n\n"
        "def checkout(user):\n    return refund(user)\n"
    )
    assert sites_for(tmp_path, caller) == []


def test_cases_carry_the_repository_the_callee_lives_in(tmp_path):
    """Without repo_root the reviewer has nowhere to look the definition up, and
    the case becomes unanswerable rather than cross-file."""
    make_repo(tmp_path)
    cases = crossfile_cases(tmp_path, eligible_files(tmp_path, min_lines=1), random.Random(0), 2)

    assert cases, "no cross-file case built from a repository that has one"
    case = cases[0]
    assert case.request.repo_root == str(tmp_path)
    assert case.kind == "defect"
    assert case.truth is not None
    assert case.request.reviewable, "a case no reviewer will look at cannot be scored"
