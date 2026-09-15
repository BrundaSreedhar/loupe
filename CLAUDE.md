# Working in this repo

A code reviewer. Four reviewers read a diff at once — security, correctness,
performance, maintainability — their findings are merged, and a separate pass
re-reads the real file and throws out the ones that don't hold up. Ships with a
test harness that breaks real code on purpose to check whether the reviewer
notices.

## Read this before spending a model call

**The Google free tier is 20 requests per day, per model.** Not 1,500 — that
number is in a lot of blog posts and it is wrong for this account. One full review
is about ten calls, so a single model buys roughly **two reviews a day**.

The cap being per model is the useful part: the quota id is
`GenerateRequestsPerDayPerProjectPerModel`, so pointing roles at different models
gives each its own allowance. `LOUPE_MODEL`, `LOUPE_VERIFIER_MODEL` and
`LOUPE_CHEAP_MODEL` are separate for exactly this reason.

Available models change often. List what this key can actually call rather than
trusting a blog post — `gemini-2.5-flash` is already retired and returns 404
pointing at `gemini-3.6-flash`.

```bash
loupe doctor          # provider, model, key, rate limit — check before running
```

`LOUPE_MODEL` applies to whichever provider is active. A model name belonging to
the other provider is detected and corrected with a warning, but set it correctly
rather than relying on that.

`LOUPE_FALLBACK_MODEL` names a second model to use once the primary's daily cap is
hit. It fires only on a confirmed daily exhaustion, never on a transient
per-minute limit. Leave it empty for eval runs: a result mixing two models
measures neither, and the harness refuses to print one.

## Commands

`./install.sh` puts `loupe` on the PATH via `uv tool install --editable`, with
settings in `~/.config/loupe/.env`. Inside this checkout `.venv/bin/loupe`
works too; they are the same code because the install is editable.

```bash
loupe local HEAD~1                  # review a local diff
loupe local --mode single           # one reviewer instead of four (1 call)
loupe local --no-verify             # skip the checking pass
loupe pr owner/repo 123             # dry run; --post writes to GitHub
loupe pipeline                         # print the compiled graph
loupe local HEAD~1 -v               # show what each stage did
loupe local HEAD~1 -vv              # add debug;  -vvv adds HTTP traffic
python -m evals.run_eval main --source ~/repo --n-defect 6 --n-clean 6
python -m evals.run_eval main --source ~/repo --n-crossfile 8   # cross-file defects
python -m evals.run_eval main --source ~/repo --dry-run    # corpus only, no calls
pytest -q
ruff check src evals tests
```

`loupe local` diffs a ref against the **working tree**, so `HEAD` means
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

## Seeing past the diff

Two mechanisms, both of which fail closed.

**Citations.** Every finding quotes the source line it reasoned from, and the
quote is matched against the file. No model call. A quote that is not there drops
the finding; a quote matching one line nearby moves the finding to it. The one
exception is load-bearing: if *no* finding in a batch carried a quote, the model
failed to fill the field and nothing is dropped — otherwise a provider-side
failure renders as a clean review, which is the mistake this repo has shipped
three times.

**Expansion.** `index.py` reads the repository once and resolves the names the
changed lines call, so a reviewer can see what `validate(payload)` actually does.
Every resolution rule prefers nothing over a guess: two definitions with one name
resolve to nothing, an import from outside the repo never falls back to a local
function of the same name, and a definition already on screen is not re-sent.
Python only, via `ast` — a regex approximating a parser across ten languages is
how a reviewer ends up reading the wrong `validate`, and being wrong there is
invisible to everyone downstream.

## What a review cost

`ReviewResult.tokens` carries calls, input, output, thinking, and cache reads
against cache writes, per model. Collected by a callback handler passed in the
graph config — not by threading a return value through the nodes — so it sees
retries and fallbacks too, and survives the parallel fan-out because LangGraph
propagates config callbacks to every branch.

Zero is not free. A provider that reports no usage must render as "not reported";
the eval harness has `Report.metered` for exactly that, and it is the same rule as
`Report.reviewed`.

## Memory

Findings carry both an `id` (uuid, per run, used to match a verdict to its claim)
and a `fingerprint` (stable across runs, used to recognise a repeat). Do not
conflate them. The fingerprint hashes normalised source around the flagged line,
never the line number — lines move on every push, and a line-keyed identity makes
every finding look new after a rebase.

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

**The progress display has two halves and they are not interchangeable.**
Spinners say what is running now and are transient; the printed lines say what
each stage did and stay. When stdout is not a terminal there is no spinner, so
the start text is printed instead — otherwise a piped log loses which reviewer
opened what. `Reporter.close()` runs in a `finally`: a spinner still turning over
a traceback is its own bug.

**A file path is diff content, in the terminal too.** Paths are
attacker-controlled on a pull request, and the report and the progress lines are
printed with Rich markup on. An unmatched `[/bold]` in a filename raises
MarkupError and takes the review down; a well-formed `[link=...]` renders as
something clickable the review never contained. Every path spliced into a markup
string goes through `rich.markup.escape` — or build a `Text` and `.append` it,
which never parses markup. Found by pointing loupe at its own diff.

**Anthropic reports cache writes in three keys, not one.** When the per-TTL
breakdown is present, `langchain_anthropic` moves the numbers into
`ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` and sets
`cache_creation` to 0 so nothing double-counts. Read only the obvious key and
every Anthropic cache write reports as zero — which reads as "warming the prefix
is free", the one conclusion the measurement exists to test. `tokens.py` sums all
three; the behaviour is in the installed package, not in anyone's memory.

**The gate needs the definitions too.** Measured, not guessed: with the verifier
given only the changed file, a correct cross-file finding was rejected as
"assumptions about the signature of code that was not provided" — its own words.
The gate is told to reject anything resting on code it cannot see, so expansion
without this makes every correct cross-file finding rejectable by construction,
and buys nothing. `prompts/definitions.py` is shared by both for that reason.

**The references belong in the cached prefix.** Same rule as the role rubric,
opposite direction: definitions from the index go into `context_message`, which is
byte-identical for all four reviewers, and `warm_cache` must send exactly what the
reviewers will send — references included. Warming without them warms nothing.

**The corpus builds cross-file cases with the reviewer's own index.** Same reason
the corpus uses the reviewer's own path filter: two implementations of "where is
this defined" drift, and then the harness measures the drift.

**`evals/seed` is frozen on purpose, and re-freezing resets every comparison.**
The generated corpus is a function of a source repo and a seed, so it moves whenever
that repo does — which means a number from last month cannot be compared with one
from today. `--seed-set` scores the committed cases instead. `freeze-set` regenerates
them, and a result against the old set and one against the new set are two different
measurements wearing the same name, so do it rarely and say so when you do.

Freeze from a repo whose source you are willing to publish. The cases embed the real
file they were cut from, and this repository has a GitHub remote.

**Two detection columns, and they mean different things.** `detection_rate` asks
only whether something was flagged within three lines of the seeded defect.
`strict_detection_rate` also requires the finding's category to match the
mutation's. Both are reported because the gap between them is how much of the
headline number is coincidence — a style reviewer objecting to a name on the same
line scores in the first column and not the second. `detection_rate` keeps its old
meaning deliberately: every recorded result was measured on it, and redefining a
column silently makes past runs incomparable.

**Parser-based mutators live in `evals/pymutations.py`, and their contract is
enforced by the decorator, not by each mutator.** `mutations.py` is line-based so one
corpus builder covers Python, TypeScript and JavaScript; `pymutations.py` gives that
up for the defects a regex cannot express — a swapped argument to a callee whose
signature was checked, an `await` dropped from a call verified to be a coroutine.
`_adapt` enforces three things: the result parses, it differs from the original, and
the reported line is the line that changed. That last one is the number the whole
harness rests on, and nothing downstream can tell a wrong ground-truth line from a
reviewer that missed the defect. The first version of that guard read
`ast.parse(mutated) is None`, which never fires — `ast.parse` raises — so the rule
was documented and not enforced. There is a test for it now.

**Do not seed a defect a linter would catch.** `ruff` runs before the reviewers and
they are told not to re-report what it found, so a mutator planting a B006 mutable
default or a bare `except` measures the pre-pass rather than the panel. Two drafted
mutators were dropped for this: `mutable_default_argument` (ruff B006) and
`widen_slice_bound`, which mostly produced non-defects — widening `skipped[:4]` in a
log message is cosmetic, and an unfair permanent miss drags detection down while
looking like the reviewer's fault.

**Evals turn the lint pre-pass off, on purpose.** The mutated file exists only in
memory. A linter reads the file on disk, which is the unmutated one, and reports
on code the reviewer was never shown. `run_review(..., lint=False)` is the switch.

**`emit` is deliberately not a graph node.** Posting to a PR is a side effect and
the eval runs the graph thousands of times. Keep it outside.

**Posting to GitHub is off by default.** `--post` is opt-in and prompts. Do not
change that default.

## Per-project config

A repo under review may carry `.loupe/` with `config.env`, `rules.md` and
`ignore`. Rules go in the **role message**, never the system prompt — putting
them in the prefix would give each reviewer a different cached prefix and defeat
the warm. They are labelled as maintainer configuration, because they sit in the
same prompt as untrusted source.

## Layout

```
src/loupe/
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
  index.py        the repo's own definitions, and what a change calls
  grounding.py    does a finding's quoted line exist, and where
  nodes/          one file per step
  prompts/        the actual prompts
  adapters/       local git, GitHub PR
evals/
  mutations.py    ways to break real code on purpose
  crossfile.py    defects only visible from a second file
  corpus.py       builds the test cases
  scoring.py      the metrics
  run_eval.py     the runner
```

## Testing

256 tests, no network. Model calls are faked at the node boundary. The end-to-end
tests in `test_graph_e2e.py` fake the models but run the real graph, which is what
catches wiring bugs the unit tests miss.

Three tests exist because they nearly went wrong silently: a planted credential
must never reach the model, a planted `ignore previous instructions` comment must
not stop a real bug being found in the same file, and a credential sitting in an
unchanged file that expansion pulls in must not leave the machine either.
