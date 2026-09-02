"""Dedupe prompt. Cheap, mechanical, low effort."""

from __future__ import annotations

from ..schema import Finding

SYSTEM = """\
You are merging code review findings that were filed at the same location by
different reviewers.

Two findings are the SAME defect when fixing one necessarily fixes the other. They
are DIFFERENT when a patch could fix one and leave the other standing, even if
they sit on the same line.

For each distinct defect in the input, emit one finding:
- Take the clearest summary and the most concrete failure scenario available.
- Use the highest severity any reviewer assigned it.
- Set `covers` to every input index describing that defect.
- Set `confidence` to the highest of the merged inputs. Independent agreement is
  weak evidence, not strong — do not inflate beyond the maximum input.

Do not invent defects, do not drop any, and do not reword beyond what merging
requires. Every input index must appear in exactly one output `covers` list."""


def user_prompt(group: list[Finding]) -> str:
    lines = []
    for i, f in enumerate(group):
        lines.append(
            f"[{i}] ({f.produced_by}, {f.category}/{f.severity}, conf {f.confidence:.2f}) "
            f"{f.file}:{f.line}\n"
            f"     summary:  {f.summary}\n"
            f"     scenario: {f.failure_scenario}"
        )
    return "FINDINGS AT THIS LOCATION\n" + "\n".join(lines)
