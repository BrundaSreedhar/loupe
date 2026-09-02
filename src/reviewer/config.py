"""Provider selection, model bindings and budgets.

Two providers are supported behind one factory. Effort is tuned per role rather
than globally: the verification pass is the precision gate and gets full effort,
the merger does mechanical work and doesn't need it. The two providers express
that differently — Anthropic through `output_config.effort`, Google through
`thinking_budget` — so the mapping lives here and nowhere else.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

# cwd first, so a repo under review can override; then the reviewer's own .env,
# because `review local --repo-root ~/other-repo` gets run from anywhere and
# python-dotenv only ever walks up from the working directory.
load_dotenv()
_own_env = Path(__file__).resolve().parents[2] / ".env"
if _own_env.is_file():
    load_dotenv(_own_env, override=False)

PROVIDER = os.getenv("REVIEWER_PROVIDER", "google").lower()

_DEFAULT_MODEL = {"google": "gemini-3.5-flash", "anthropic": "claude-opus-5"}
_CHEAP_MODEL = {"google": "gemini-3.5-flash-lite", "anthropic": "claude-opus-5"}
_KEY_ENV = {"google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

if PROVIDER not in _DEFAULT_MODEL:
    raise ValueError(f"REVIEWER_PROVIDER must be one of {sorted(_DEFAULT_MODEL)}, got {PROVIDER!r}")

MODEL = os.getenv("REVIEWER_MODEL") or _DEFAULT_MODEL[PROVIDER]
CHEAP_MODEL = os.getenv("REVIEWER_CHEAP_MODEL") or _CHEAP_MODEL[PROVIDER]

# Requests per minute, client-side. Google's free tier is measured in tens of RPM,
# and one multi-agent review is ~8-12 calls, so without this an eval run trips the
# quota within the first minute. 0 disables the limiter.
RPM = int(os.getenv("REVIEWER_RPM", "10" if PROVIDER == "google" else "0"))

# Gemini thinking budgets: -1 lets the model decide, 0 turns thinking off,
# a positive integer caps it. Some models reject 0 — override if yours does.
THINKING_HIGH = int(os.getenv("REVIEWER_THINKING_HIGH", "-1"))
THINKING_LOW = int(os.getenv("REVIEWER_THINKING_LOW", "0"))

# Context budget, in tokens, for a single file's source window.
FILE_TOKEN_BUDGET = int(os.getenv("REVIEWER_FILE_TOKEN_BUDGET", "12000"))
# Total across all files in one review. Past this, files are dropped by rank
# and the report says so rather than quietly reviewing half the change.
REVIEW_TOKEN_BUDGET = int(os.getenv("REVIEWER_REVIEW_TOKEN_BUDGET", "120000"))
# Lines of context kept either side of a hunk when a file doesn't fit whole.
WINDOW_PADDING = int(os.getenv("REVIEWER_WINDOW_PADDING", "40"))
# Findings shown. A review with 8 findings gets read; one with 40 gets closed.
MAX_REPORTED = int(os.getenv("REVIEWER_MAX_REPORTED", "12"))
# Two findings this close on the same file are candidates for merging.
MERGE_LINE_WINDOW = int(os.getenv("REVIEWER_MERGE_LINE_WINDOW", "3"))

SPECIALIST_ROLES = ("security", "correctness", "performance", "maintainability")


def credentials_present() -> bool:
    if PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    return bool(os.getenv("GOOGLE_API_KEY"))


def key_env_var() -> str:
    return _KEY_ENV[PROVIDER]


@cache
def _rate_limiter() -> InMemoryRateLimiter | None:
    if RPM <= 0:
        return None
    # max_bucket_size 1 means no burst: the fan-out cannot fire four calls at once
    # and immediately spend a minute's quota.
    return InMemoryRateLimiter(requests_per_second=RPM / 60, max_bucket_size=1)


def _llm(effort: str, max_tokens: int = 16000, cheap: bool = False) -> BaseChatModel:
    model = CHEAP_MODEL if cheap else MODEL

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
    return _llm("high", max_tokens=8000)


@cache
def merger_llm() -> BaseChatModel:
    """Mechanical: decide whether two findings describe the same defect."""
    return _llm("low", max_tokens=4000, cheap=True)
