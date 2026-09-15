"""The parser-based Python mutators.

Ground truth in this harness is a line number, and nothing downstream can tell a
wrong one from a reviewer that missed the defect. So these tests assert the three
rules `pymutations` claims to enforce — the result parses, it differs, and the
reported line is the line that changed — over every mutator, rather than trusting
each one to have got it right.
"""

from __future__ import annotations

import ast
import random

import pytest

from evals.corpus import build_request
from evals.pymutations import all_python_mutators

# Written to give every mutator a candidate site: a verified two-parameter callee,
# a coroutine called with await, an `and` in a condition, a multi-type except, and
# a trailing re-raise.
SAMPLE = '''\
import asyncio


def charge(account, amount):
    return account - amount


async def settle(account):
    await asyncio.sleep(0)
    return account


async def run(account, amount, ledger):
    if account is not None and amount > 0:
        balance = charge(account, amount)
        settled = await settle(balance)
        try:
            ledger.write(settled)
        except (ValueError, TypeError) as exc:
            ledger.log(exc)
            raise
        return settled
    return None
'''

ALL = sorted(all_python_mutators())


def _apply(name: str, source: str, seed: int = 0):
    fn = all_python_mutators()[name]
    lines = source.splitlines()
    mutation = fn(lines, random.Random(seed))
    return ("\n".join(lines), mutation)


@pytest.mark.parametrize("name", ALL)
def test_the_result_is_still_valid_python(name):
    """A syntactically broken file tests whether the reviewer notices a
    SyntaxError, which every reviewer does. It is not a case."""
    mutated, mutation = _apply(name, SAMPLE)
    if mutation is None:
        pytest.skip(f"{name} found no candidate site in the sample")
    ast.parse(mutated)


@pytest.mark.parametrize("name", ALL)
def test_the_result_actually_differs(name):
    """A mutation that changed nothing is a clean control mislabelled as a defect.
    It scores as a miss forever and reads as the reviewer's fault."""
    mutated, mutation = _apply(name, SAMPLE)
    if mutation is None:
        pytest.skip(f"{name} found no candidate site in the sample")
    assert mutated != SAMPLE


@pytest.mark.parametrize("name", ALL)
def test_the_reported_line_is_the_line_that_changed(name):
    """The number the whole harness rests on. A deletion reports a parenthesised
    note instead of replacement text, following `mutations.remove_guard`."""
    mutated, mutation = _apply(name, SAMPLE)
    if mutation is None:
        pytest.skip(f"{name} found no candidate site in the sample")
    assert 1 <= mutation.line <= len(mutated.splitlines())
    if mutation.after.startswith("("):
        assert mutation.before in SAMPLE
    else:
        assert mutated.splitlines()[mutation.line - 1].strip() == mutation.after


@pytest.mark.parametrize("name", ALL)
def test_the_reported_line_shows_up_as_changed_in_the_diff(name):
    """Recall is scored against a line the reviewer was shown. A ground-truth line
    outside the diff measures nothing."""
    mutated, mutation = _apply(name, SAMPLE)
    if mutation is None:
        pytest.skip(f"{name} found no candidate site in the sample")
    request = build_request("sample.py", SAMPLE, mutated, "test")
    changed = request.files[0].changed_lines
    # A deletion has no line of its own in the new file; the hunk covers where it was.
    assert changed, f"{name} produced a diff with no changed lines"
    if not mutation.after.startswith("("):
        assert mutation.line in changed


@pytest.mark.parametrize("name", ALL)
def test_every_mutator_is_restricted_to_python(name):
    """Read by `corpus.attempt`. Without it these spend attempts on TypeScript
    files they cannot parse, and the corpus quietly comes back short."""
    assert all_python_mutators()[name].suffixes == (".py", ".pyi")


@pytest.mark.parametrize("name", ALL)
def test_nothing_fires_on_source_that_is_not_python(name):
    ts = "export function charge(account: number, amount: number) { return account - amount; }\n"
    _, mutation = _apply(name, ts)
    assert mutation is None


# ─── the individual defects ─────────────────────────────────────────────────


def test_swapped_arguments_keeps_the_call_valid_but_wrong():
    mutated, mutation = _apply("swap_call_arguments", SAMPLE)
    assert mutation is not None
    assert mutation.category == "correctness"
    # Same arguments, different order — no crash, wrong answer.
    assert sorted(mutation.before) == sorted(mutation.after)
    assert mutation.before != mutation.after


def test_dropping_an_argument_names_the_signature_it_no_longer_matches():
    """The description is what a human reads when a case looks like a false
    positive. "Call lost an argument" alone does not settle it; the parameter list
    does."""
    _, mutation = _apply("drop_call_argument", SAMPLE)
    assert mutation is not None
    assert "charge" in mutation.description
    assert "account, amount" in mutation.description


def test_the_await_is_only_dropped_from_a_verified_coroutine():
    mutated, mutation = _apply("unawaited_async_call", SAMPLE)
    assert mutation is not None
    assert "await" in mutation.before and "await" not in mutation.after
    # `settle` is an `async def` in this same file — that is the check the regex
    # version in mutations.py cannot make.
    assert "settle" in mutation.description
    tree = ast.parse(mutated)
    assert any(
        isinstance(n, ast.AsyncFunctionDef) and n.name == "settle" for n in ast.walk(tree)
    )


def test_a_narrowed_except_names_the_type_that_now_escapes():
    _, mutation = _apply("narrow_exception_catch", SAMPLE)
    assert mutation is not None
    assert "TypeError" in mutation.description
    assert "TypeError" not in mutation.after


def test_a_swallowed_exception_leaves_the_logging_behind():
    """The handler still looks as though it deals with the failure, which is what
    makes this shape survive review."""
    mutated, mutation = _apply("swallow_exception", SAMPLE)
    assert mutation is not None
    assert mutation.after == "(re-raise deleted)"
    assert "ledger.log(exc)" in mutated
    assert "raise" not in mutated.split("except (")[1]


def test_an_inverted_condition_flips_only_one_operator():
    _, mutation = _apply("invert_bool_operator", SAMPLE)
    assert mutation is not None
    assert " and " in mutation.before and " or " in mutation.after


# ─── the guard around all of them ───────────────────────────────────────────


def test_a_mutator_that_breaks_the_syntax_produces_no_case():
    """The decorator's job, tested directly rather than hoped for. A mutator that
    emits invalid Python must contribute nothing, not a broken case."""
    from evals.mutations import Mutation
    from evals.pymutations import _adapt

    @_adapt
    def wrecker(source, rng):
        return "def broken(:\n", Mutation("wrecker", "correctness", 1, "a", "b", "d")

    lines = SAMPLE.splitlines()
    original = list(lines)
    assert wrecker(lines, random.Random(0)) is None
    # Compared as a list: the mutators write back through `lines[:]`, and joining
    # it drops SAMPLE's trailing newline, which would fail for the wrong reason.
    assert lines == original, "the source must be left untouched"


def test_a_mutator_that_changes_nothing_produces_no_case():
    from evals.mutations import Mutation
    from evals.pymutations import _adapt

    @_adapt
    def no_op(source, rng):
        return source, Mutation("no_op", "correctness", 1, "a", "b", "d")

    assert no_op(SAMPLE.splitlines(), random.Random(0)) is None


def test_a_mutator_that_raises_produces_no_case():
    """Unusual source must not take the whole corpus build down."""
    from evals.pymutations import _adapt

    @_adapt
    def explodes(source, rng):
        raise ValueError("unusual source")

    assert explodes(SAMPLE.splitlines(), random.Random(0)) is None
