"""Provider selection, model bindings and budgets.

Two providers are supported behind one factory. Effort is tuned per role rather
than globally: the verification pass is the precision gate and gets full effort,
the merger does mechanical work and doesn't need it. The two providers express
that differently — Anthropic through `output_config.effort`, Google through
`thinking_budget` — so the mapping lives here and nowhere else.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

# Three places, most specific first. Nothing overrides anything already set, so
# an env var in the shell always wins.
#
#   0. .loupe/config.env in the current directory or above — a project's own
#      settings, travelling with the checkout.
#   1. cwd and above — a plain .env, same idea.
#   2. the reviewer's own checkout — for `--repo-root ~/elsewhere`, since
#      python-dotenv only ever walks up from the working directory. Present only
#      for an editable install; a normal install puts __file__ in site-packages.
#   3. ~/.config/loupe/.env — the one that makes `loupe` work from anywhere
#      once it is installed as a command rather than run out of the checkout.
USER_CONFIG = Path(
    os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")
) / "reviewer" / ".env"

from .project import find_dir as _find_project_dir  # noqa: E402

_project = _find_project_dir(Path.cwd())
if _project is not None and (_project / "config.env").is_file():
    load_dotenv(_project / "config.env", override=False)

load_dotenv()
for _candidate in (Path(__file__).resolve().parents[2] / ".env", USER_CONFIG):
    if _candidate.is_file():
        load_dotenv(_candidate, override=False)

PROVIDER = os.getenv("LOUPE_PROVIDER", "google").lower()

_DEFAULT_MODEL = {
    "google": "gemini-3.5-flash",
    "anthropic": "claude-opus-5",
    "ollama": "qwen2:7b",
}
_CHEAP_MODEL = {
    "google": "gemini-3.5-flash-lite",
    "anthropic": "claude-opus-5",
    "ollama": "qwen2:7b",
}
_KEY_ENV = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "(none — runs locally)",
}

# A local model to fall back to when the day's quota is exhausted. Empty disables
# it, and the review fails instead — which is the right default for anything whose
# output you intend to trust or measure.
FALLBACK_MODEL = os.getenv("LOUPE_FALLBACK_MODEL", "")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Local models are slower per call but have no per-minute cap.
OLLAMA_NUM_CTX = int(os.getenv("LOUPE_OLLAMA_NUM_CTX", "16384"))

if PROVIDER not in _DEFAULT_MODEL:
    raise ValueError(f"LOUPE_PROVIDER must be one of {sorted(_DEFAULT_MODEL)}, got {PROVIDER!r}")

# Model names are recognisable by prefix, which is enough to catch the common
# misconfiguration: LOUPE_MODEL left set to one provider's model while
# LOUPE_PROVIDER points at another. Without this, switching to a local model
# silently asks Ollama for "gemini-3.5-flash" and fails somewhere much later.
_MODEL_PREFIX = {"google": ("gemini",), "anthropic": ("claude",)}


def _belongs_to_another_provider(name: str) -> str | None:
    for provider, prefixes in _MODEL_PREFIX.items():
        if provider != PROVIDER and name.lower().startswith(prefixes):
            return provider
    return None


def _resolve_model(env_var: str, defaults: dict[str, str]) -> str:
    configured = os.getenv(env_var)
    if not configured:
        return defaults[PROVIDER]
    if (other := _belongs_to_another_provider(configured)) is not None:
        logging.getLogger(__name__).warning(
            "%s=%r looks like a %s model but LOUPE_PROVIDER=%s; using %r instead. "
            "Set %s explicitly if that was deliberate.",
            env_var, configured, other, PROVIDER, defaults[PROVIDER], env_var,
        )
        return defaults[PROVIDER]
    return configured


MODEL = _resolve_model("LOUPE_MODEL", _DEFAULT_MODEL)
CHEAP_MODEL = _resolve_model("LOUPE_CHEAP_MODEL", _CHEAP_MODEL)

# Google's free-tier daily cap is per project *and per model* — the quota id in
# the 429 body is GenerateRequestsPerDayPerProjectPerModel. So pointing different
# roles at different models gives each its own daily allowance instead of all
# three draining one. Defaults to MODEL, i.e. no change unless you opt in.
VERIFIER_MODEL = _resolve_model("LOUPE_VERIFIER_MODEL", dict.fromkeys(_DEFAULT_MODEL, MODEL))

# Requests per minute, client-side. Google's free tier is measured in tens of RPM,
# and one multi-agent review is ~8-12 calls, so without this an eval run trips the
# quota within the first minute. 0 disables the limiter.
RPM = int(os.getenv("LOUPE_RPM", "10" if PROVIDER == "google" else "0"))

# Gemini thinking budgets: -1 lets the model decide, 0 turns thinking off,
# a positive integer caps it. Some models reject 0 — override if yours does.
THINKING_HIGH = int(os.getenv("LOUPE_THINKING_HIGH", "-1"))
THINKING_LOW = int(os.getenv("LOUPE_THINKING_LOW", "0"))

# Context budget, in tokens, for a single file's source window.
FILE_TOKEN_BUDGET = int(os.getenv("LOUPE_FILE_TOKEN_BUDGET", "12000"))
# Total across all files in one review. Past this, files are dropped by rank
# and the report says so rather than quietly reviewing half the change.
REVIEW_TOKEN_BUDGET = int(os.getenv("LOUPE_REVIEW_TOKEN_BUDGET", "120000"))
# Lines of context kept either side of a hunk when a file doesn't fit whole.
WINDOW_PADDING = int(os.getenv("LOUPE_WINDOW_PADDING", "40"))
# Findings shown. A review with 8 findings gets read; one with 40 gets closed.
MAX_REPORTED = int(os.getenv("LOUPE_MAX_REPORTED", "12"))
# How many findings to verify, as a multiple of what will be reported. Verifying
# more than this is paying to judge findings that ranking discards unseen.
VERIFY_HEADROOM = int(os.getenv("LOUPE_VERIFY_HEADROOM", "2"))
# Two findings this close on the same file are candidates for merging.
MERGE_LINE_WINDOW = int(os.getenv("LOUPE_MERGE_LINE_WINDOW", "3"))

_ALL_ROLES = ("security", "correctness", "performance", "maintainability")

# Which reviewers to run. Each is one model call, so this is the largest single
# lever on cost: dropping to two halves the fan-out. Whether the dropped two were
# earning their calls is a question for the harness, not for taste.
SPECIALIST_ROLES = tuple(
    r.strip() for r in os.getenv("LOUPE_ROLES", ",".join(_ALL_ROLES)).split(",") if r.strip()
)
_unknown = set(SPECIALIST_ROLES) - set(_ALL_ROLES)
if _unknown:
    raise ValueError(f"LOUPE_ROLES has unknown role(s): {sorted(_unknown)}")

# "parallel" runs one call per reviewer. "combined" runs a single call carrying
# every rubric — four times cheaper, and it gives up the independence that makes
# four reviewers worth having. An arm to measure, not a default to assume.
FANOUT = os.getenv("LOUPE_FANOUT", "parallel").lower()

# Whether to run the repository's own linters before reviewing.
#   auto — on for a local diff, off for a pull request. Running a repo's tooling
#          executes its config, which is fine for your code and not fine for a
#          branch someone else wrote.
#   on   — always. off — never.
LINT = os.getenv("LOUPE_LINT", "auto").lower()


def lint_enabled(source: str) -> bool:
    if LINT == "on":
        return True
    if LINT == "off":
        return False
    return source == "local"

# What to do when a credential is found in the diff.
#   redact — blank the value, review the rest (default: the secret never leaves,
#            and you still get a review)
#   block  — refuse the whole review
#   warn   — report it and send anyway. Never a good idea on a free tier whose
#            terms permit training on inputs.
ON_SECRET = os.getenv("LOUPE_ON_SECRET", "redact").lower()


def credentials_present() -> bool:
    if PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if PROVIDER == "ollama":
        return True  # nothing to authenticate against
    return bool(os.getenv("GOOGLE_API_KEY"))


def key_env_var() -> str:
    return _KEY_ENV[PROVIDER]


# Burst allowance. The fan-out is genuinely concurrent, so a bucket of 1 turns
# four parallel reviewers into four serial ones spaced 60/RPM apart — the average
# rate is respected either way, but the latency is four times worse for nothing.
BURST = int(os.getenv("LOUPE_BURST", str(len(SPECIALIST_ROLES))))

# Warming a prefix costs one blocking call before the fan-out can start. Below
# this size the cache saves less than the extra round-trip costs.
WARM_MIN_TOKENS = int(os.getenv("LOUPE_WARM_MIN_TOKENS", "4000"))

# How many verification samples a *borderline* finding gets. 1 disables it.
# Sampling everything three times costs 2N extra calls for almost no gain: a
# verifier that confirms with clear reasoning does not change its mind on
# resample. The close calls are where the extra look pays for itself.
CONSENSUS_SAMPLES = int(os.getenv("LOUPE_CONSENSUS_SAMPLES", "1"))

# A finding is borderline when the two independent signals disagree: the reviewer
# was unsure and the gate confirmed anyway, or the reviewer was confident and the
# gate rejected. Agreement in either direction needs no second opinion.
CONSENSUS_LOW = float(os.getenv("LOUPE_CONSENSUS_LOW", "0.6"))
CONSENSUS_HIGH = float(os.getenv("LOUPE_CONSENSUS_HIGH", "0.85"))

# "per_file" verifies all of a file's findings in one call, "per_finding" uses one
# call each. Per-file is far cheaper; per-finding keeps the judgements independent.
VERIFY_MODE = os.getenv("LOUPE_VERIFY_MODE", "per_file")


@cache
def _rate_limiter() -> InMemoryRateLimiter | None:
    if RPM <= 0 or PROVIDER == "ollama":
        return None  # a local model has no quota to protect
    return InMemoryRateLimiter(
        requests_per_second=RPM / 60, max_bucket_size=max(BURST, 1)
    )


def _ollama(model: str, max_tokens: int) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=OLLAMA_URL,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=max_tokens,
        temperature=0,
    )


def local_llm(max_tokens: int = 8000) -> BaseChatModel | None:
    """The fallback model, or None when no fallback is configured."""
    if not FALLBACK_MODEL:
        return None
    return _ollama(FALLBACK_MODEL, max_tokens)


def _llm(
    effort: str, max_tokens: int = 16000, cheap: bool = False, model: str | None = None
) -> BaseChatModel:
    model = model or (CHEAP_MODEL if cheap else MODEL)

    if PROVIDER == "ollama":
        return _ollama(model, max_tokens)

    if PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            rate_limiter=_rate_limiter(),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        max_output_tokens=max_tokens,
        thinking_budget=THINKING_HIGH if effort == "high" else THINKING_LOW,
        rate_limiter=_rate_limiter(),
    )


@cache
def specialist_llm() -> BaseChatModel:
    """One binding shared by all four roles — the rubric differs, not the model."""
    return _llm("high")


@cache
def verifier_llm() -> BaseChatModel:
    """The verification pass. Runs at high effort — lowering it trades directly
    against the precision the gate exists to provide."""
    return _llm("high", max_tokens=8000, model=VERIFIER_MODEL)


@cache
def merger_llm() -> BaseChatModel:
    """Mechanical: decide whether two findings describe the same defect."""
    return _llm("low", max_tokens=4000, cheap=True)


def with_short_output(llm: Any, limit: int = 16) -> Any:
    """Cap the reply length, using whatever the bound provider calls that.

    Anthropic takes `max_tokens`, Gemini `max_output_tokens`, Ollama
    `num_predict`. Binding the wrong one is not a soft failure — the value is
    forwarded to the provider's request config, which rejects unknown keys.
    """
    key = {
        "anthropic": "max_tokens",
        "google": "max_output_tokens",
        "ollama": "num_predict",
    }[PROVIDER]
    return llm.bind(**{key: limit})


def structured(llm: Any, schema: Any, label: str) -> Any:
    """Bind a schema, and wrap in a local fallback when one is configured.

    Applied after `with_structured_output` rather than before, because the
    fallback has to produce the same shape as the primary — a fallback that
    returns free text where the caller expects a parsed object is not a fallback.
    """
    primary = llm.with_structured_output(schema, method="json_schema")
    local = local_llm()
    if local is None or PROVIDER == "ollama":
        return primary
    from .fallback import LocalFallback

    return LocalFallback(primary, local.with_structured_output(schema, method="json_schema"), label)
