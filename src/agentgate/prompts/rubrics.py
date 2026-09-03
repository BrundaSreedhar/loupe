"""Per-role rubrics.

The non-goals matter more than the goals. A reviewer told only what to look for
will find it everywhere; the explicit exclusions are most of what keeps four
panellists from filing the same observation four times in four vocabularies.
"""

from __future__ import annotations

RUBRICS: dict[str, dict[str, str]] = {
    "security": {
        "looks_for": """\
- Injection: SQL/NoSQL built by concatenation, shell invocation with untrusted
  input, template injection, path traversal in file operations.
- Authorization gaps: a handler that reads or mutates a resource without checking
  the caller owns it; IDOR; a permission check that runs after the side effect.
- Secrets: credentials or tokens hardcoded, logged, or returned in a response.
- Unsafe deserialization: pickle, YAML unsafe load, eval/exec on external input.
- SSRF and open redirects: user-controlled URLs fetched or redirected to.
- Crypto misuse: weak or homemade algorithms, static IVs/salts, comparison of
  secrets with ==, missing signature verification.
- Missing validation at a trust boundary — the request edge, not internal calls.""",
        "non_goals": """\
- Correctness bugs with no security consequence (the correctness reviewer has them).
- Performance, resource use, or test coverage.
- Defence-in-depth suggestions where no concrete attack exists. "An attacker could
  conceivably" is not a finding.""",
    },
    "correctness": {
        "looks_for": """\
- Logic errors: inverted conditions, wrong operator, off-by-one, wrong variable.
- Boundary cases the change newly reaches: empty collection, None/null, zero,
  negative, single element, maximum length.
- Error handling: swallowed exceptions, catching too broadly, recovery that leaves
  inconsistent state, error paths that return success.
- Concurrency: shared mutable state without synchronisation, check-then-act races,
  await/lock ordering, non-atomic read-modify-write.
- Resource lifetime: unclosed handles, use-after-close, leaks on the error path.
- Async: dropped awaits, unawaited coroutines, fire-and-forget that loses errors.
- Contract violations: returning a different shape or type than callers expect.""",
        "non_goals": """\
- Security-specific defects (the security reviewer has them).
- Performance, unless the code is outright non-terminating.
- Test coverage, naming, structure.""",
    },
    "performance": {
        "looks_for": """\
- A query, network call, or file read inside a loop — the N+1 shape.
- Accidental quadratic behaviour: nested iteration over the same growing input,
  repeated `in` against a list, string concatenation in a loop.
- Work repeated per call that could be computed once.
- Unbounded results: a fetch with no limit or pagination that grows with the data.
- Blocking I/O or CPU-bound work on an async or request-handling path.
- Large copies of data where a reference or generator would do.""",
        "non_goals": """\
- Micro-optimisation with no evidence the path is hot.
- Anything whose cost is constant and small.
- Correctness, security, tests, style.
- Speculation about scale the code will never see. If the collection is bounded
  by a config constant, it is not a performance defect.""",
    },
    "maintainability": {
        "looks_for": """\
- New logic branches with no corresponding test, when a test suite plainly exists.
- Changes to a public signature, return shape, or exported name that would break
  existing callers.
- Code introduced by this change that is already unreachable or unused.
- Errors raised or logged without enough information to act on them.
- Logic duplicated from somewhere else visible in the given context.""",
        "non_goals": """\
- Style, naming, formatting, import order, comment density. Never report these.
- Subjective architecture preference. "I would have used a class" is not a finding.
- Anything the security, correctness, or performance reviewers cover.
- File length, function length, or any other metric on its own.

Your bar is higher than the other reviewers'. This category produces the most
noise across the industry. If you are not sure, report nothing.""",
    },
    "generalist": {
        "looks_for": """\
- Any defect in the change: security, correctness, performance, or a missing test
  for newly added logic.
- Prioritise defects that would actually break something over anything cosmetic.""",
        "non_goals": """\
- Style, naming, formatting, comment density, import order.
- Restatements of what the code does.
- Subjective architecture preference.""",
    },
}
