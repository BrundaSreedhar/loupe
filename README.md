# loupe

A code reviewer that is quiet unless it has something to say.

A jeweller's loupe magnifies one small area and shows the flaws that are
invisible at normal scale. This does that to a diff.

Four specialists read a change in parallel, each looking for a different class of
defect. Everything they produce is merged, then re-checked against the full source
by a separate pass that is told to reject. What survives is what you see.

```bash
./install.sh
loupe doctor
loupe local HEAD~1
```

## Architecture

```
                            ┌──────────────┐
   diff ─────────────────►  │  preflight   │  secrets never leave the machine;
   local git or GitHub PR   └──────┬───────┘  source is marked as data, not
                                   │          instructions
                            ┌──────▼───────┐
                            │   prepare    │  budgeted, line-numbered windows
                            └──────┬───────┘  of each changed file
                                   │
                            ┌──────▼───────┐
                            │  warm_cache  │  one write, so the four reviewers
                            └──────┬───────┘  below read a shared prefix
                                   │
         ┌───────────┬─────────────┼─────────────┬───────────┐
         │           │             │             │           │
    ┌────▼────┐ ┌────▼─────┐ ┌─────▼──────┐ ┌────▼────────┐  │  Send() × 4
    │security │ │correct-  │ │performance │ │maintain-    │  │  in parallel
    │         │ │ness      │ │            │ │ability      │  │
    └────┬────┘ └────┬─────┘ └─────┬──────┘ └────┬────────┘  │
         │           │             │             │           │
         └───────────┴─────────────┼─────────────┴───────────┘
                                   │  fan-in via a reducer, so no branch is lost
                            ┌──────▼───────┐
                            │    dedupe    │  merge before verifying — cheaper
                            └──────┬───────┘
                                   │
                            ┌──────▼───────┐
                            │    verify    │  ◄── the gate. Batched per file,
                            └──────┬───────┘      reads whole source, told to
                                   │              reject rather than agree
                            ┌──────▼───────┐
                            │   finalize   │  apply verdicts, rank, truncate
                            └──────┬───────┘
                                   │
                    terminal ◄─────┴─────► GitHub review
                                             (dry run unless --post)
```

`loupe pipeline` prints the compiled version, which is the authoritative one.

Three ordering decisions carry weight:

- **Dedupe before verify.** Verification costs a call per file; every duplicate
  removed first is a call not made.
- **Warm the cache before the fan-out.** All four reviewers share one large
  prefix. Fired cold in parallel they all miss it and all pay to write it.
- **The role rubric is the last message, not the system prompt.** Caching matches
  on a prefix, so anything role-specific placed before the source gives every
  branch a different prefix and defeats the warm entirely.

`emit` is deliberately not a graph node. Posting to a pull request is a side
effect, and the eval harness runs this graph thousands of times.

## Using it

```bash
loupe local HEAD~1           # a local diff
loupe local --staged         # the index
loupe local --mode single    # one generalist instead of four
loupe local --no-verify      # skip the gate, for measurement
loupe local HEAD~1 -v        # show what each stage did

loupe pr owner/repo 123      # a pull request — dry run
loupe pr owner/repo 123 --post   # ...and publish, after confirming
```

`loupe local <ref>` diffs a ref against your working tree, so `HEAD` means
uncommitted changes and `HEAD~1` means the last commit plus anything uncommitted.

Findings carry a concrete failure scenario, and a replacement patch where the
reviewer could write one exactly — which becomes a one-click suggestion on a pull
request. Docs, lockfiles, generated and vendored files are skipped, and the run
says what it skipped.

## Memory

A second review of the same branch reports what changed, not the same list again:

```
  1 fixed since last review   ·   3 still open from before
```

Findings are identified by a hash of the normalised code around them rather than
by line number, so an edit twenty lines above a defect does not make it a new
finding, and reformatting does not turn one open finding into a resolved one plus
a new one. Fixing the defect does change the identity, which is the point.

State lives in `~/.config/loupe/state`, never in the repository — a generated file
in the working tree would end up in the next diff. `--fresh` ignores it.

## Per-project settings

```bash
cd your-project
loupe init
```

```
.loupe/
  rules.md      review guidance for this codebase
  ignore        extra paths to skip, one glob per line
  config.env    settings, read before the global config
```

`rules.md` is the one worth writing. Generic reviewers find generic bugs; the
defects a codebase produces repeatedly are the ones only its maintainers can name:

```markdown
- All database access goes through `db/gateway.ts`. A direct `pg.query` call is a
  high-severity finding even when the SQL itself is safe.
- Money is always integer cents. A float touching a currency value is a bug.
```

Configuration is read from `.loupe/config.env`, then a nearby `.env`, then
`~/.config/loupe/.env`. A variable set in your shell always wins.

## Measuring it

The harness breaks real files from a real repository in known ways, and makes
semantics-preserving changes to others. Recall is mechanical ground truth; every
finding on an unchanged-behaviour file is a false positive.

```bash
python -m evals.run_eval main --source ~/some-repo --dry-run    # corpus only
python -m evals.run_eval main --source ~/some-repo --arms all --repeats 3
```

Arms vary two switches — four specialists or one generalist, gate on or off — so
each contribution can be read separately:

| comparison | what it shows |
|---|---|
| `multi` vs `single` | what four specialists find that one generalist doesn't |
| `+verify` vs bare | what the gate removes, and what recall it costs |

Detection rate only means something next to the false-positive column: a reviewer
that flags every line scores 100% on detection alone. `--repeats` reports the
spread, which is the noise floor under every other number.

The harness refuses to print results from a run that reviewed nothing, or that
mixed two models — both have happened, and both looked like clean measurements.

## Providers and tracing

Runs on Google Gemini or Anthropic Claude, selected with `LOUPE_PROVIDER`. A
second model can be named as a fallback for when the primary's daily allowance
runs out — the cap is per model, so that is a whole extra allowance. Any review
that used it says so, because a result mixing two models measures neither.

Set `LANGSMITH_TRACING=true` and a key and one review becomes one trace, with
named spans per reviewer and a finding id on every verification.

See `.env.example` for the full set of settings.

## Development

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
pytest -q
ruff check src evals tests
```

112 tests, no network — model calls are faked at the node boundary. The
end-to-end tests run the real graph against fake models, which is what catches
wiring bugs the unit tests miss.

## What it will not do

- **Judge whether a change is a good idea.** It can tell you code is broken. It
  cannot tell you the feature is wrong.
- **Know anything outside the repository.** Environment settings, feature flags
  and other services' expectations are not in the files it reads.
- **Handle deeply tangled control flow.** Rare paths through complicated code are
  where it is weakest, and where the expensive bugs live. True of every tool in
  this category.
- **Work across many repositories.** This is a single-repo tool.
