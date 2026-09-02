"""Eval runner.

Scores a reviewer configuration against a corpus of seeded defects and clean
controls. Arms vary two switches — `multi`/`single` reviewers and the verification
gate on/off — so the contribution of each can be read separately.

    python -m evals.run_eval --source ~/some-repo --arms all
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from reviewer.config import MODEL, PROVIDER, RPM
from reviewer.runner import run_review
from reviewer.schema import ReviewRequest

from .corpus import Case, build
from .scoring import Report, score_case, spread

app = typer.Typer(add_completion=False)
console = Console()

ARMS: dict[str, tuple[str, bool]] = {
    "multi+verify": ("multi", True),
    "multi": ("multi", False),
    "single+verify": ("single", True),
    "single": ("single", False),
}

# Rough model calls per review, per arm: warm + reviewers + merges + verifications.
# Deliberately on the high side — the point is to catch a run that will blow a
# daily quota before it starts, not to predict spend to the call.
CALLS_PER_REVIEW: dict[str, int] = {
    "multi+verify": 11,
    "multi": 6,
    "single+verify": 4,
    "single": 1,
}


def estimate_calls(selected: list[str], n_cases: int, repeats: int) -> int:
    return sum(CALLS_PER_REVIEW[a] for a in selected) * n_cases * repeats


def run_arm(cases: list[Case], mode: str, verify: bool, workers: int) -> Report:
    report = Report()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_review,
                case.request,
                mode=mode,
                verify=verify,
                run_name=f"eval:{mode}{'+v' if verify else ''}:{case.id}",
            ): case
            for case in cases
        }
        for fut in as_completed(futures):
            case = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 — one bad case must not
                # abort an eval run that costs real money.
                console.print(f"[red]{case.id} failed:[/red] {type(exc).__name__}: {exc}")
                continue
            report.scores.append(score_case(case, result))
    return report


def render_comparison(results: dict[str, list[Report]]) -> None:
    table = Table(title="Arm comparison", title_style="bold", header_style="dim")
    table.add_column("arm")
    table.add_column("detection", justify="right")
    table.add_column("FP / clean diff", justify="right")
    table.add_column("silent on clean", justify="right")
    table.add_column("gate rejected", justify="right")
    table.add_column("merged away", justify="right")

    for arm, reps in results.items():
        det_m, det_s = spread([r.detection_rate for r in reps])
        fp_m, fp_s = spread([r.fp_per_clean for r in reps])
        sil_m, _ = spread([r.clean_silence_rate for r in reps])
        rej_m, _ = spread([r.rejection_rate for r in reps])
        mrg_m, _ = spread([r.merge_rate for r in reps])
        pm = len(reps) > 1
        table.add_row(
            arm,
            f"{det_m:.0%}" + (f" ±{det_s:.0%}" if pm else ""),
            f"{fp_m:.2f}" + (f" ±{fp_s:.2f}" if pm else ""),
            f"{sil_m:.0%}",
            f"{rej_m:.0%}" if "verify" in arm else "—",
            f"{mrg_m:.0%}",
        )
    console.print()
    console.print(table)
    console.print()


@app.command()
def main(
    source: Path = typer.Option(..., "--source", help="Repo to draw corpus source from."),
    arms: str = typer.Option("multi+verify", "--arms", help="Comma-separated, or 'all'."),
    n_defect: int = typer.Option(20, "--n-defect"),
    n_clean: int = typer.Option(20, "--n-clean"),
    repeats: int = typer.Option(1, "--repeats", help="Repeat runs to measure the noise floor."),
    workers: int = typer.Option(4, "--workers"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("evals/results"), "--out"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build the corpus and stop."),
) -> None:
    selected = list(ARMS) if arms == "all" else [a.strip() for a in arms.split(",")]
    unknown = [a for a in selected if a not in ARMS]
    if unknown:
        raise typer.BadParameter(f"Unknown arm(s): {unknown}. Choose from {list(ARMS)}.")

    cases = build(source.expanduser().resolve(), n_defect=n_defect, n_clean=n_clean, seed=seed)
    n_d = sum(1 for c in cases if c.kind == "defect")
    n_c = len(cases) - n_d
    console.print(f"[dim]Corpus: {n_d} seeded defects, {n_c} clean controls, from {source}[/dim]")

    if n_d < n_defect or n_c < n_clean:
        console.print(
            f"[yellow]Note:[/yellow] asked for {n_defect}/{n_clean}, got {n_d}/{n_c} — "
            "mutators found fewer candidate sites than requested in this source tree."
        )

    if dry_run:
        by_mutator: dict[str, int] = {}
        for c in cases:
            key = c.truth.name if c.truth else "clean"
            by_mutator[key] = by_mutator.get(key, 0) + 1
        console.print(json.dumps(by_mutator, indent=2))
        return

    total = len(selected) * repeats * len(cases)
    calls = estimate_calls(selected, len(cases), repeats)
    console.print(
        f"[dim]{len(selected)} arm(s) x {repeats} repeat(s) = {total} reviews "
        f"· ~{calls} model calls · {PROVIDER}/{MODEL}[/dim]"
    )
    if RPM:
        console.print(
            f"[dim]At {RPM} req/min that is roughly {calls / RPM / 60:.1f} hour(s) "
            "of wall clock.[/dim]"
        )
    if calls > 400:
        console.print(
            f"[yellow]~{calls} calls.[/yellow] Free tiers are typically capped in the "
            "low thousands per day and far lower per minute — check your own limits "
            "before committing to this. Shrink with --n-defect/--n-clean, fewer "
            "--arms, or --repeats 1."
        )
        typer.confirm("Continue?", abort=True)

    results: dict[str, list[Report]] = {}
    for arm in selected:
        mode, verify = ARMS[arm]
        reps: list[Report] = []
        for i in range(repeats):
            console.print(f"[dim]  {arm} — run {i + 1}/{repeats}[/dim]")
            reps.append(run_arm(cases, mode, verify, workers))
        results[arm] = reps

    render_comparison(results)

    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"eval-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "seed": seed,
                "n_defect": n_d,
                "n_clean": n_c,
                "arms": {
                    arm: {
                        "runs": [r.as_dict() for r in reps],
                        "by_mutator": reps[0].by_mutator(),
                    }
                    for arm, reps in results.items()
                },
            },
            indent=2,
        )
    )
    console.print(f"[dim]Written to {path}[/dim]")


@app.command()
def push(
    source: Path = typer.Option(..., "--source"),
    dataset: str = typer.Option("code-review-seeded-bugs", "--dataset"),
    n_defect: int = typer.Option(20, "--n-defect"),
    n_clean: int = typer.Option(20, "--n-clean"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Push the corpus to LangSmith as a dataset, so experiments are comparable there."""
    from langsmith import Client

    if not os.getenv("LANGSMITH_API_KEY"):
        raise typer.BadParameter("LANGSMITH_API_KEY is not set.")

    cases = build(source.expanduser().resolve(), n_defect=n_defect, n_clean=n_clean, seed=seed)
    client = Client()
    if client.has_dataset(dataset_name=dataset):
        ds = client.read_dataset(dataset_name=dataset)
    else:
        ds = client.create_dataset(
            dataset_name=dataset,
            description="Seeded code defects plus semantics-preserving controls.",
        )

    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"case_id": c.id, "request": c.request.model_dump(mode="json")},
                "outputs": {
                    "kind": c.kind,
                    "line": c.truth.line if c.truth else None,
                    "mutator": c.truth.name if c.truth else None,
                },
            }
            for c in cases
        ],
    )
    console.print(f"[green]Pushed {len(cases)} examples to '{dataset}'.[/green]")


def langsmith_target(inputs: dict) -> dict:
    """Target for `client.evaluate` against the pushed dataset."""
    request = ReviewRequest.model_validate(inputs["request"])
    result = run_review(request, mode="multi", verify=True)
    return {
        "findings": [
            {"file": f.file, "line": f.line, "summary": f.summary} for f in result.accepted
        ]
    }


def detected_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Recall on seeded defects. Mechanical — no judge, no drift."""
    if reference_outputs.get("kind") != "defect":
        return {"key": "detected", "score": None}
    target = reference_outputs["line"]
    hit = any(abs(f["line"] - target) <= 3 for f in outputs.get("findings", []))
    return {"key": "detected", "score": int(hit)}


def noise_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """False positives on semantics-preserving controls."""
    if reference_outputs.get("kind") != "clean":
        return {"key": "false_positives", "score": None}
    return {"key": "false_positives", "score": len(outputs.get("findings", []))}


if __name__ == "__main__":
    app()
