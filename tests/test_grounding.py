"""The citation check.

A reviewer quotes the line its claim rests on; the quote is matched against the
real file. These cover the four things that decide whether that helps or hurts:
a quote that is right, a quote that is invented, a quote that is a line or two
out, and a quote that is too common to place.
"""

from __future__ import annotations

from loupe.grounding import apply, check, cited_text
from loupe.schema import Finding

SOURCE = "\n".join(
    [
        "def total(items):",              # 1
        "    amount = 0",                 # 2
        "    for i in range(len(items)):",  # 3
        "        amount += items[i].price",  # 4
        "    return amount",              # 5
        "",                               # 6
        "def label(item):",               # 7
        "    amount = 0",                 # 8
        "    n = 0",                      # 9
        "    return amount",              # 10
    ]
)


def finding(line: int, evidence: str, file: str = "a.py") -> Finding:
    return Finding(
        file=file,
        line=line,
        category="correctness",
        severity="high",
        summary="s",
        failure_scenario="concrete inputs produce a wrong total",
        confidence=0.8,
        evidence=evidence,
        id="f1",
        produced_by="correctness",
    )


def test_a_quote_at_the_line_it_claims_passes():
    assert check("        amount += items[i].price", SOURCE, 4, window=5).ok


def test_whitespace_and_spacing_do_not_matter():
    """A formatter reflowing a line must not invalidate a real finding."""
    assert check("amount+=items[ i ].price", SOURCE, 4, window=5).ok


def test_the_listing_prefix_is_stripped():
    """Models copy the `>123| ` decoration about half the time."""
    assert check(">  4|         amount += items[i].price", SOURCE, 4, window=5).ok


def test_a_quote_that_is_not_in_the_file_fails():
    result = check("    amount += items[i].prices  # tax", SOURCE, 4, window=5)
    assert not result.ok
    assert result.reason == "mismatch"


def test_a_finding_is_moved_to_the_line_it_quoted():
    """Reviewers are routinely a line or two out. The quote says where the code is."""
    result = check("    for i in range(len(items)):", SOURCE, 4, window=5)
    assert result.ok and result.moved
    assert result.line == 3


def test_a_quote_matching_several_nearby_lines_is_not_placed():
    """`amount = 0` is written twice. Matching one of them is not evidence of
    which one the reviewer read."""
    result = check("    amount = 0", SOURCE, 4, window=5)
    assert not result.ok
    assert result.reason == "ambiguous"


def test_a_quote_too_short_to_carry_information_is_not_moved():
    """`n = 0` could be any line in any file. It is accepted where it is claimed
    and never used to relocate a finding."""
    result = check("    n = 0", SOURCE, 4, window=5)
    assert not result.ok
    assert result.reason == "trivial"
    assert check("    n = 0", SOURCE, 9, window=5).ok


def test_a_match_outside_the_window_is_not_reached_for():
    result = check("def label(item):", SOURCE, 1, window=2)
    assert not result.ok
    assert result.reason == "mismatch"


def test_multi_line_quotes_match_as_a_block():
    quote = "for i in range(len(items)):\n    amount += items[i].price"
    assert check(quote, SOURCE, 3, window=5).ok


def test_no_source_means_no_verdict_either_way():
    """An unreadable file is not evidence that a finding is wrong."""
    assert check("anything at all", "", 4, window=5).ok


def test_apply_drops_the_invented_and_keeps_the_real():
    good = finding(4, "amount += items[i].price")
    bad = finding(4, "amount += items[i].price * tax_rate")
    outcome = apply([good, bad], {"a.py": SOURCE}, window=5)

    assert [f.evidence for f in outcome.kept] == [good.evidence]
    assert outcome.dropped == 1
    assert "a.py:4" in outcome.reasons[0]


def test_apply_re_anchors_a_finding_to_its_quote():
    outcome = apply(
        [finding(4, "    for i in range(len(items)):")], {"a.py": SOURCE}, window=5
    )
    assert outcome.moved == 1
    assert outcome.kept[0].line == 3


def test_no_citations_at_all_drops_nothing():
    """The failure this project has shipped three times: a stage that fails and
    reports the failure as a clean result. A model that filled no citations has
    told us nothing about the findings, so nothing may be discarded on it."""
    findings = [finding(4, ""), finding(5, "")]
    outcome = apply(findings, {"a.py": SOURCE}, window=5)

    assert outcome.kept == findings
    assert outcome.skipped
    assert outcome.missing == 2
    assert outcome.dropped == 0


def test_a_missing_citation_among_real_ones_is_dropped():
    """Once the model has shown it can fill the field, an empty one is a failure
    to cite rather than a provider that does not support the field."""
    outcome = apply(
        [finding(4, "amount += items[i].price"), finding(5, "")], {"a.py": SOURCE}, window=5
    )
    assert not outcome.skipped
    assert outcome.dropped == 1
    assert outcome.missing == 1


def test_cited_text_keeps_indentation_for_display():
    """Matching is whitespace-insensitive; the report is not — the reader is
    looking at code."""
    quoted = cited_text(">  4|         amount += items[i].price")
    assert quoted == "        amount += items[i].price"
    assert cited_text("```\n    x = 1\n```") == "    x = 1"


# ─── citations through a merge ──────────────────────────────────────────────


def _merged(line: int):
    from loupe.schema import MergedFinding

    return MergedFinding(
        file="a.py",
        line=line,
        category="correctness",
        severity="high",
        summary="merged",
        failure_scenario="concrete inputs produce a wrong total",
        confidence=0.9,
        covers=[0, 1],
    )


def test_a_merged_finding_inherits_the_citation_it_was_built_from():
    """The merger reads summaries, not source, so it has no citation to give."""
    from loupe.nodes.dedupe import rejoin

    kept = finding(4, "amount += items[i].price")
    out = rejoin(_merged(4), [kept, finding(4, "amount += items[i].price")], SOURCE)

    assert out.evidence == kept.evidence
    assert out.merged_from == [kept.id, kept.id]


def test_a_merge_cannot_move_a_finding_away_from_its_quote():
    """The line came from a model. The quote was checked against the file."""
    from loupe.nodes.dedupe import rejoin

    kept = finding(4, "amount += items[i].price")
    out = rejoin(_merged(2), [kept, kept], SOURCE)

    assert out.line == 4


def test_a_merged_finding_can_still_be_recognised_next_time():
    """Merged findings used to come out with an empty fingerprint, so memory
    treated every one of them as new on every run."""
    from loupe.nodes.dedupe import rejoin

    kept = finding(4, "amount += items[i].price")
    out = rejoin(_merged(4), [kept, kept], SOURCE)

    assert out.fingerprint
