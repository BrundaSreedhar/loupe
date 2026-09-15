"""Metrics.

Detection rate and the clean-case false-positive rate are always reported
together: a reviewer that flags every line scores 100% detection, so neither
number constrains the reviewer without the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from loupe.schema import ReviewResult

from .corpus import Case

ANCHOR_TOLERANCE = 3


@dataclass
class CaseScore:
    case_id: str
    kind: str
    detected: bool | None  # None for clean cases
    false_positives: int
    raw: int
    merged: int
    accepted: int
    rejection_rate: float
    # Flagged the right line *and* filed it under the category the mutation
    # actually belongs to. Separate from `detected` rather than replacing it, and
    # defaulted rather than required: every recorded result was measured on the
    # looser rule, so changing what that column means would make past runs
    # incomparable, and every existing caller keeps working.
    detected_category: bool | None = None
    contexts: int = 0
    fallback_stages: int = 0
    mutator: str = ""
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0


@dataclass
class Report:
    scores: list[CaseScore] = field(default_factory=list)

    def _d(self) -> list[CaseScore]:
        return [s for s in self.scores if s.kind == "defect"]

    def _c(self) -> list[CaseScore]:
        return [s for s in self.scores if s.kind == "clean"]

    @property
    def detection_rate(self) -> float:
        d = self._d()
        return sum(1 for s in d if s.detected) / len(d) if d else 0.0

    @property
    def strict_detection_rate(self) -> float:
        """Detection that also got the reason right.

        `detection_rate` asks only whether something was flagged near the seeded
        line, so a style reviewer objecting to a variable name three lines away
        scores as a catch. The gap between the two columns is how much of the
        headline number is luck.
        """
        d = self._d()
        return sum(1 for s in d if s.detected_category) / len(d) if d else 0.0

    @property
    def fp_per_clean(self) -> float:
        c = self._c()
        return mean([s.false_positives for s in c]) if c else 0.0

    @property
    def clean_silence_rate(self) -> float:
        """Share of clean cases with zero findings — the number a developer feels."""
        c = self._c()
        return sum(1 for s in c if s.false_positives == 0) / len(c) if c else 0.0

    @property
    def merge_rate(self) -> float:
        vals = [1 - (s.merged / s.raw) for s in self.scores if s.raw]
        return mean(vals) if vals else 0.0

    @property
    def rejection_rate(self) -> float:
        vals = [s.rejection_rate for s in self.scores if s.merged]
        return mean(vals) if vals else 0.0

    @property
    def tokens_per_review(self) -> float:
        """Input plus output, averaged over the cases that actually ran.

        Reported beside detection because that is the trade being made. An arm
        that finds one more defect for twice the tokens is a different proposition
        from one that finds it for free, and the table used to show only half of
        that."""
        return mean([s.tokens_in + s.tokens_out for s in self.scores]) if self.scores else 0.0

    @property
    def calls_per_review(self) -> float:
        return mean([s.calls for s in self.scores]) if self.scores else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache. The warm exists to make this
        large in `multi`; in `single` there is nobody to share a prefix with, so a
        low number there is correct rather than a regression."""
        total = sum(s.tokens_in for s in self.scores)
        return sum(s.cache_read for s in self.scores) / total if total else 0.0

    @property
    def metered(self) -> int:
        """Cases whose token usage was actually reported by the provider.

        Zero means the numbers below are missing, not free — the same trap as
        reporting a review that never happened as a clean one."""
        return sum(1 for s in self.scores if s.calls)

    @property
    def reviewed(self) -> int:
        """Cases where the reviewers were actually shown something.

        Without this, a pipeline that shows the reviewer nothing scores a
        confident 0% detection and 0.00 false positives — which reads like a
        cautious reviewer and is in fact a broken one."""
        return sum(1 for s in self.scores if s.contexts > 0)

    @property
    def fell_back(self) -> int:
        """Cases where some stage ran on the local fallback model. A score mixing
        two models measures neither of them."""
        return sum(1 for s in self.scores if s.fallback_stages)

    def by_mutator(self) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for s in self._d():
            buckets.setdefault(s.mutator, []).append(bool(s.detected))
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}

    def as_dict(self) -> dict[str, float]:
        return {
            "detection_rate": self.detection_rate,
            "strict_detection_rate": self.strict_detection_rate,
            "fp_per_clean": self.fp_per_clean,
            "clean_silence_rate": self.clean_silence_rate,
            "merge_rate": self.merge_rate,
            "rejection_rate": self.rejection_rate,
            "n_defect": len(self._d()),
            "n_clean": len(self._c()),
            "reviewed": self.reviewed,
            "tokens_per_review": self.tokens_per_review,
            "calls_per_review": self.calls_per_review,
            "cache_hit_rate": self.cache_hit_rate,
            "metered": self.metered,
            "fell_back": self.fell_back,
            "scored": len(self.scores),
        }


def score_case(case: Case, result: ReviewResult) -> CaseScore:
    accepted = result.accepted
    detected: bool | None = None
    fp = 0

    detected_category: bool | None = None

    if case.kind == "defect" and case.truth is not None:
        target = case.truth.line
        path = case.request.files[0].path
        on_target = [
            f
            for f in accepted
            if f.file == path and abs(f.line - target) <= ANCHOR_TOLERANCE
        ]
        detected = bool(on_target)
        detected_category = any(f.category == case.truth.category for f in on_target)
        # Findings elsewhere in a defect case are not scored: the file genuinely
        # may contain other real problems we never labelled.
    else:
        fp = len(accepted)

    return CaseScore(
        case_id=case.id,
        kind=case.kind,
        detected=detected,
        detected_category=detected_category,
        false_positives=fp,
        raw=int(result.usage.get("raw_count", 0)),
        merged=int(result.usage.get("merged_count", 0)),
        accepted=len(accepted),
        rejection_rate=result.usage.get("rejection_rate", 0.0),
        contexts=int(result.usage.get("contexts", 0)),
        fallback_stages=int(result.usage.get("fallback_stages", 0)),
        mutator=case.truth.name if case.truth else "",
        calls=result.tokens.calls,
        tokens_in=result.tokens.input,
        tokens_out=result.tokens.output,
        cache_read=result.tokens.cache_read,
    )


def spread(values: list[float]) -> tuple[float, float]:
    """Mean and population sd across repeat runs — the noise floor under every
    other number in the report."""
    return (mean(values), pstdev(values)) if values else (0.0, 0.0)
