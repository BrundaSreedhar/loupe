# Multi-agent code reviewer

Four specialist reviewers — security, correctness, performance, maintainability —
fan out over one diff in parallel. Their findings are merged, then each one is
re-checked against the full source file by a separate verification pass before it
reaches a human.

Ships with an eval harness that scores the reviewer against seeded defects and
clean controls, and a `single` mode that runs one generalist reviewer for
comparison. Runs on Gemini or Claude.

## Install

```bash
./install.sh          # makes `review` available from any directory
review doctor         # check configuration
```

That uses `uv tool install --editable`, which builds an isolated environment for
the tool and links `review` into `~/.local/bin` — so its dependencies never
collide with whatever project you are standing in. Editable means edits to this
checkout take effect without reinstalling.

Settings live in `~/.config/reviewer/.env`, created from `.env.example` on first
install. A globally installed command cannot rely on a `.env` in the checkout, so
that file is what makes it work from anywhere. Three locations are read, most
specific first, and a shell variable always wins:

1. `.env` in the current directory or above — lets a repo carry its own settings
2. `.env` in this checkout — only present for an editable install
3. `~/.config/reviewer/.env` — the one a normal install uses

To uninstall: `uv tool uninstall multi-agent-code-reviewer`.

## Develop

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
cp .env.example .env      # add GOOGLE_API_KEY (or ANTHROPIC_API_KEY)
review doctor             # check what is configured before spending anything
review local HEAD~1
```

## Providers

Runs on Google Gemini or Anthropic Claude, selected with `REVIEWER_PROVIDER`.
Gemini is the default because AI Studio has a free tier; get a key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

|  | `google` | `anthropic` |
|---|---|---|
| default model | `gemini-3.5-flash` | `claude-opus-5` |
| key | `GOOGLE_API_KEY` | `ANTHROPIC_API_KEY` |
| effort control | `thinking_budget` (-1 auto, 0 off) | `output_config.effort` |
| prefix caching | implicit | explicit `cache_control` |

The prompt structure is provider-independent — the shared context is kept
byte-identical across reviewers and the per-role rubric goes last, which is what
makes a prefix cache hit on either backend. Only the way the breakpoint is
*expressed* differs, and that is confined to `cached_block()`.

### Free-tier notes

Google's free tier is Flash-only, is capped per minute and per day, and those
caps change — check yours at
[aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit) rather
than trusting a number written here. `REVIEWER_RPM` throttles the client side to
match; the fan-out will otherwise fire four calls at once.

Reporting at the time of writing is that free-tier prompts and responses may be
used to improve Google's models. This tool sends your source code to the API, so
read Google's current terms before pointing it at anything you would not publish.
The paid tier and Anthropic do not carry that condition.

## The graph

```
START → prepare → warm_cache ──┬─→ security ──────┐
                               ├─→ correctness ───┤
                               ├─→ performance ───┤   Annotated[list, add]
                               └─→ maintainability┘   fan-in
                                                  ↓
                                              dedupe
                                                  ↓  Send() per finding
                                              verify   ← the gate
                                                  ↓
                                             finalize → END
```

`review graph` prints the compiled mermaid, which is the authoritative version.

Three ordering decisions carry weight:

- **Dedupe runs before verify.** Verification is one model call per finding, so
  every duplicate removed first is a call not made.
- **warm_cache runs before the fan-out.** All four reviewers share one large
  context prefix. Fired cold in parallel they all miss and all pay to write it
  (~4×1.25× input); warming once makes it one write plus three reads (~1.55×).
- **The role rubric is the last message, not the system prompt.** Caching is a
  prefix match over `tools → system → messages`, so anything role-specific placed
  before the context would give every branch a different prefix and defeat the
  warm entirely.

`emit` is deliberately *not* a graph node. Posting to a PR is a side effect, and
keeping it outside means the eval harness can run thousands of reviews with no
possibility of writing to anyone's repository.

## Reviewing

```bash
review local HEAD~1                  # local diff
review local --staged                # the index
review local --mode single           # one generalist, the baseline
review local --no-verify             # skip the gate (for measurement)
review pr owner/repo 123             # a GitHub PR — dry run
review pr owner/repo 123 --post      # ...and actually post, after confirming
```

`--post` is off by default and prompts before writing. Findings are posted as one
batched review with inline comments, not N separate comments.

## Evaluating

The corpus is built by mutating real source from a real repository. Each mutation
records the line it broke, so recall is mechanical ground truth rather than a
hand-label that drifts. The clean half applies semantics-preserving edits, where
every finding is by definition a false positive.

```bash
python -m evals.run_eval main --source ~/some-repo --dry-run     # inspect the corpus
python -m evals.run_eval main --source ~/some-repo --n-defect 6 --n-clean 6
python -m evals.run_eval main --source ~/some-repo --arms all --repeats 3
python -m evals.run_eval push --source ~/some-repo               # → LangSmith dataset
```

The runner estimates model calls and wall-clock time before starting, and asks
for confirmation past ~400 calls. On a free tier, start small: `--arms all` over
40 cases is well over a thousand calls.

Arms are `multi+verify`, `multi`, `single+verify`, `single`:

| Comparison | What it shows |
|---|---|
| `multi` vs `single` | What four specialists find that one generalist doesn't |
| `+verify` vs bare | What the gate removes, and what recall it costs |

Reported metrics: detection rate on seeded defects, false positives per clean
diff, share of clean diffs reviewed in silence, gate rejection rate, and merge
rate. `--repeats` gives the ± spread, which is the noise floor under every other
number — a 4-point difference means nothing against ±5 points of run-to-run
variance.

Detection rate is only meaningful next to the clean-diff column — a reviewer that
flags every line scores 100% on detection alone.

## Latency

On a rate-limited tier, wall-clock is bound by call count, not by tokens or
compute. One `multi+verify` review is roughly `1 warm + 4 reviewers + merges +
verifications`, and at 10 req/min every call is 6 seconds of waiting before the
model does anything.

Three things keep that down:

- **The rate limiter allows a burst** (`REVIEWER_BURST`, default 4). With a bucket
  of 1 the four concurrent reviewers queue behind each other and the fan-out buys
  nothing.
- **Cache warming is skipped below `REVIEWER_WARM_MIN_TOKENS`** (default 4000).
  It is a blocking call the fan-out waits on, so on a small review it costs a full
  round-trip to save less than one.
- **Verification is batched per file** (`REVIEWER_VERIFY_MODE=per_file`). One call
  per file instead of one per finding, with the source sent once rather than once
  per finding. Set `per_finding` to keep judgements strictly independent — the
  eval harness can price the difference.

Knobs worth checking before blaming the design: `REVIEWER_RPM` should match your
account's actual limit, `REVIEWER_THINKING_LOW/HIGH` control Gemini thinking
budgets (0 is off, -1 is dynamic), and `REVIEWER_CHEAP_MODEL` can point the
merger at Flash-Lite.

## Observability

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` and one review becomes one
trace. Spans are named and tagged so the trace answers questions worth asking:
`specialist:security`, `verify:path:line`, with `finding_id` on every verify span
so a bad verdict is traceable back to the claim that produced it.

## Known weaknesses

- **Reviewer and verifier are the same model family**, so collusion is a live
  risk. The gate is prompted adversarially and given the full file rather than the
  diff so it reasons from different evidence, but a rejection rate near 0% means
  it has gone to sleep — watch that number.
- **The clean corpus is synthetic.** Semantics-preserving mutations are not the
  same distribution as real refactors. Mixing in real commits from your own
  history is the fix, and is not implemented yet.
- **Mutator coverage is uneven** across languages — the SQL and exception
  mutators find few sites in a TypeScript tree. `by_mutator` in the report breaks
  detection down per defect type so a skewed corpus is visible rather than hidden.
