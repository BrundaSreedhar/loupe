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
                            │    expand    │  the definitions those lines call,
                            └──────┬───────┘  looked up in the repository
                                   │
                            ┌──────▼───────┐
                            │     lint     │  the repo's own tools go first, and
                            └──────┬───────┘  what they found is not re-reported
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
                                   │  each finding quotes the line it rests on;
                                   │  quotes that are not in the file are dropped
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

Four ordering decisions carry weight:

- **Dedupe before verify.** Verification costs a call per file; every duplicate
  removed first is a call not made.
- **Warm the cache before the fan-out.** All four reviewers share one large
  prefix. Fired cold in parallel they all miss it and all pay to write it.
- **Expand before warming.** The definitions pulled in from the repository are
  part of that shared prefix. Assembled after the warm, they change the prefix,
  and nothing hits the cache.
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

Findings carry a concrete failure scenario, the source line the claim rests on,
and a replacement patch where the reviewer could write one exactly — which becomes
a one-click suggestion on a pull request. Docs, lockfiles, generated and vendored
files are skipped, and the run says what it skipped.

## Beyond the diff

Two things stop a reviewer from having to guess.

**Every finding quotes the line it rests on**, copied from the file. The quote is
matched against the real source, which costs nothing — no model call — and drops
the findings that describe code nobody wrote. It also fixes anchoring: a quote
that matches one line a little further down moves the finding there, rather than
throwing away a real defect over a line number. The quote is printed with the
finding, so the claim and its evidence arrive together.

**Definitions the changed lines call are looked up and included.** A reviewer
shown only the file you edited has no idea what `validate(payload)` does, so it
either guesses — which is where a share of the false alarms come from — or says
nothing and misses a real bug.

The rule everywhere in that lookup is *show nothing rather than the wrong thing*.
A reviewer handed the wrong `validate` is fluent and confident and wrong, and
nobody downstream can tell. So a name defined in two places is skipped, a name
imported from outside the repository is never answered with a local function that
happens to share it, and a definition already on screen is not sent twice.

It is Python only, through the standard library's parser. Other languages are
still reviewed; they get no expansion. A regex standing in for a parser across ten
languages resolves the wrong `validate` eventually, which is the one outcome this
is built to avoid. Definitions go through the same secret scan as the diff — they
come from files your change never touched.

```bash
loupe local HEAD~1                # expansion on, it is a local checkout
LOUPE_INDEX=off loupe local HEAD~1   # diff only
LOUPE_GROUNDING=off loupe local HEAD~1   # keep findings whose quote is wrong
```

Expansion is off for pull requests by default: that repository is not on this
disk, so there is nothing to read.

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
python -m evals.run_eval main --source ~/some-repo --n-crossfile 8
```

`--n-crossfile` seeds defects that are invisible in the file that contains them:
a call whose arguments stop matching a function defined in another file. They are
the measurement for repository expansion, and the only honest one — if showing a
reviewer what its change calls is worth the tokens, detection on these goes up and
detection elsewhere does not move. Run it twice, once with `LOUPE_INDEX=off`. Each
result file records which switches were on, because a number that does not say
what produced it cannot be compared with anything.

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

Runs on Google Gemini, Anthropic Claude, or a local OpenAI-compatible endpoint,
selected with `LOUPE_PROVIDER`. A
second model can be named as a fallback for when the primary's daily allowance
runs out — the cap is per model, so that is a whole extra allowance. Any review
that used it says so, because a result mixing two models measures neither.

Loupe has three explicit execution modes:

| mode | model endpoint | tracing | GitHub PR command |
|---|---|---|---|
| `cloud` (default) | Gemini or Claude | opt-in | available |
| `private` | the endpoint you configure | off by default | available |
| `offline` | loopback local runtime only | forcibly off | unavailable |

For an air-gapped local review, serve an OpenAI-compatible model with Ollama,
vLLM, or llama.cpp and configure:

```bash
LOUPE_MODE=offline LOUPE_PROVIDER=local loupe local HEAD~1
```

The local endpoint defaults to `http://127.0.0.1:11434/v1`. `loupe doctor
--egress` lists every destination Loupe may contact and what it can receive.
Set `LOUPE_TRACING=on` and a LangSmith key only when you explicitly want a
review trace; tracing is otherwise disabled.

For CI and coding agents, append `--output json` to `loupe local` or `loupe pr`
for a structured result rather than terminal panels.

See `.env.example` for the full set of settings.

## Development

```bash
uv venv --python 3.13 && uv pip install -e ".[dev]"
pytest -q
ruff check src evals tests
```

222 tests, no network — model calls are faked at the node boundary. The
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
