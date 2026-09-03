# Working in this repo

A code reviewer. Four reviewers read a diff at once — security, correctness,
performance, maintainability — their findings are merged, and a separate pass
re-reads the real file and throws out the ones that don't hold up. Ships with a
test harness that breaks real code on purpose to check whether the reviewer
notices.

## Read this before spending a model call

**The Google free tier is 20 requests per day.** Not 1,500 — that number is in a
lot of blog posts and it is wrong for this account. One full review is about ten
calls, so the daily budget is roughly **two reviews**.

```bash
review doctor          # provider, model, key, rate limit — check before running
```

Ollama is installed locally with `qwen2:7b`. Use a local model for anything that
is testing the plumbing rather than the review quality. Every bug found in this
repo so far has been structural — a bad corpus, two filters disagreeing, a broken
guard — and a small local model would have caught all of them for free.

Switch providers with `REVIEWER_PROVIDER=google|anthropic`. Nothing else changes.

## Commands

```bash
review local HEAD~1                  # review a local diff
review local --mode single           # one reviewer instead of four (1 call)
review local --no-verify             # skip the checking pass
review pr owner/repo 123             # dry run; --post writes to GitHub
review graph                         # print the compiled graph
review local HEAD~1 -v               # show what each stage did
review local HEAD~1 -vv              # add debug;  -vvv adds HTTP traffic
python -m evals.run_eval main --source ~/repo --n-defect 6 --n-clean 6
python -m evals.run_eval main --source ~/repo --dry-run    # corpus only, no calls
pytest -q
ruff check src evals tests
```

`review local` diffs a ref against the **working tree**, so `HEAD` means
uncommitted changes and `HEAD~1` means the last commit plus anything uncommitted.

## How to work here

**Check the installed package, don't recall it.** Model IDs, library APIs and
free-tier limits in this space go stale in months. `pip show`, read the source in
`.venv`, or fetch the docs. Several things in this repo were built wrong the first
time from a confident memory.

**Plain English in documentation and comments.** No word a reader would have to
look up, unless it is explained in the same sentence. This applies to README,
comments, and anything written for a person.

**Describe what the code does; don't assert what the project is.** No mission
statements, no "X is the product", no arguing the case for a design decision in a
file that is supposed to explain it. A comment saying why an ordering is
load-bearing is useful. A paragraph declaring the project's thesis is not.

**Talk through the approach before building a substantial new module**, especially
when a real tool already exists for the job. Hand-rolling an approximation of
something that has a mature implementation is usually the wrong call.

**A handled error still has to be visible.** Every stage catches its own
failures so one bad step cannot kill a review — which is also how a review that
lost most of its work came to look exactly like a clean one. Anything caught goes
into the `problems` channel (reduced with `operator.add`, like the others, because
stages fail concurrently) and is printed at the end. Never catch something and
only log it.

**Never report a failure as a measurement.** The eval harness has done this three
times: an empty corpus, a filter that rejected everything, and a guard that
skipped when there was nothing to check. Every zero it prints must be provably
"we looked and found nothing" rather than "we never looked". `Report.reviewed`
exists for this; keep it honest.

## Things that will bite

**One filter, not two.** `filters.is_reviewable_path` decides what gets reviewed,
and `evals/corpus.py` uses the same function. They diverged once and the entire
test corpus turned out to be `numpy` and `onnxruntime` source, scored as 0%
detection. Do not add a second skip list.

**Prompt caching is a prefix match.** The shared system prompt and the file
contents must be byte-identical across all four reviewers, so the per-role rubric
is the **last** message, not the system prompt. Moving it back into the system
prompt silently gives every reviewer a different prefix and kills the cache.

**The state reducers are load-bearing.** `findings` and `verdicts` use
`Annotated[list, operator.add]` because several nodes write them at once. Without
the reducer, last-write-wins throws away three of the four reviewers' work.

**429 is two different errors.** Per-minute means retry; per-day means stop.
Google sends a short `retryDelay` even for a daily cap, so honouring it blindly
retries all day. `quota.py` tells them apart — use `should_retry` in retry
policies and let `DailyQuotaExhausted` propagate rather than catching it.

**Broad excepts must let quota errors through.** `verify` rejects findings it
cannot check, which is right for a bad response and wrong for a quota failure —
that produces a clean-looking review that never happened. Call `raise_if_terminal`
first.

**`emit` is deliberately not a graph node.** Posting to a PR is a side effect and
the eval runs the graph thousands of times. Keep it outside.

**Posting to GitHub is off by default.** `--post` is opt-in and prompts. Do not
change that default.

## Layout

```
src/reviewer/
  cli.py          commands
  runner.py       one entry point; the CLI and the eval both use it
  graph.py        node wiring
  state.py        graph state and the reducers
  schema.py       what a finding is
  config.py       provider, model, budgets, rate limits
  filters.py      what is worth reviewing
  safety.py       secret and injection scanning
  quota.py        429 classification
  context.py      building what a reviewer sees
  nodes/          one file per step
  prompts/        the actual prompts
  adapters/       local git, GitHub PR
evals/
  mutations.py    ways to break real code on purpose
  corpus.py       builds the test cases
  scoring.py      the metrics
  run_eval.py     the runner
```

## Testing

76 tests, no network. Model calls are faked at the node boundary. The end-to-end
tests in `test_graph_e2e.py` fake the models but run the real graph, which is what
catches wiring bugs the unit tests miss.

Two tests exist because they nearly went wrong silently: a planted credential must
never reach the model, and a planted `ignore previous instructions` comment must
not stop a real bug being found in the same file.
