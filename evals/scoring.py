"""Metrics.

Detection rate and the clean-case false-positive rate are always reported
together: a reviewer that flags every line scores 100% detection, so neither
number constrains the reviewer without the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from reviewer.schema import ReviewResult

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
    contexts: int = 0
    mutator: str = ""


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
    def reviewed(self) -> int:
        """Cases where the reviewers were actually shown something.

        Without this, a pipeline that shows the reviewer nothing scores a
        confident 0% detection and 0.00 false positives — which reads like a
        cautious reviewer and is in fact a broken one."""
        return sum(1 for s in self.scores if s.contexts > 0)

    def by_mutator(self) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for s in self._d():
            buckets.setdefault(s.mutator, []).append(bool(s.detected))
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}

    def as_dict(self) -> dict[str, float]:
        return {
            "detection_rate": self.detection_rate,
            "fp_per_clean": self.fp_per_clean,
            "clean_silence_rate": self.clean_silence_rate,
            "merge_rate": self.merge_rate,
            "rejection_rate": self.rejection_rate,
            "n_defect": len(self._d()),
            "n_clean": len(self._c()),
            "reviewed": self.reviewed,
            "scored": len(self.scores),
        }


def score_case(case: Case, result: ReviewResult) -> CaseScore:
    accepted = result.accepted
    detected: bool | None = None
    fp = 0

    if case.kind == "defect" and case.truth is not None:
        target = case.truth.line
        detected = any(
            f.file == case.request.files[0].path and abs(f.line - target) <= ANCHOR_TOLERANCE
            for f in accepted
        )
        # Findings elsewhere in a defect case are not scored: the file genuinely
        # may contain other real problems we never labelled.
    else:
        fp = len(accepted)

    return CaseScore(
        case_id=case.id,
        kind=case.kind,
        detected=detected,
        false_positives=fp,
        raw=int(result.usage.get("raw_count", 0)),
        merged=int(result.usage.get("merged_count", 0)),
        accepted=len(accepted),
        rejection_rate=result.usage.get("rejection_rate", 0.0),
        contexts=int(result.usage.get("contexts", 0)),
        mutator=case.truth.name if case.truth else "",
    )


def spread(values: list[float]) -> tuple[float, float]:
    """Mean and population sd across repeat runs — the noise floor under every
    other number in the report."""
    return (mean(values), pstdev(values)) if values else (0.0, 0.0)
