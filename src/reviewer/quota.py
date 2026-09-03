"""Telling the two kinds of 429 apart.

A 429 means one of two opposite things, and treating them alike is how a run
wastes twenty minutes achieving nothing:

  per-minute  — you went too fast. Wait the delay the server names and continue;
                the work will succeed.
  per-day     — you are out of quota until the window resets. Every retry is
                guaranteed to fail, and the default retry policy will cheerfully
                make three of them per node before giving up, for every remaining
                case in an eval run.

Google says which in the error body: a `quotaId` containing "PerDay" versus
"PerMinute", plus a `RetryInfo.retryDelay`. This module reads that and turns a
daily exhaustion into a distinct exception the caller can abort on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Scope = Literal["per_minute", "per_day", "unknown"]

_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")
_QUOTA_ID = re.compile(r"'quotaId':\s*'([^']+)'")
_QUOTA_VALUE = re.compile(r"'quotaValue':\s*'(\d+)'")


class DailyQuotaExhausted(RuntimeError):
    """Out of requests for the day. Retrying cannot help."""


@dataclass(frozen=True)
class RateLimit:
    scope: Scope
    retry_after: float | None
    quota_id: str | None
    quota_value: int | None

    @property
    def retryable(self) -> bool:
        # "unknown" is treated as retryable: a transient limit retried needlessly
        # costs seconds, while a daily limit misread as transient costs the run.
        # Only a confirmed daily cap is worth refusing outright.
        return self.scope != "per_day"

    def describe(self) -> str:
        if self.scope == "per_day":
            cap = f" (cap: {self.quota_value}/day)" if self.quota_value else ""
            return f"daily quota exhausted{cap} — retrying will not help"
        if self.retry_after:
            return f"per-minute limit — server asked for {self.retry_after:.0f}s"
        return "rate limited"


def classify(exc: BaseException) -> RateLimit | None:
    """Return rate-limit details, or None if this is not a rate-limit error."""
    if not is_rate_limit(exc):
        return None

    text = str(exc)
    quota_id = (m := _QUOTA_ID.search(text)) and m.group(1)
    delay = (m := _RETRY_DELAY.search(text)) and float(m.group(1))
    value = (m := _QUOTA_VALUE.search(text)) and int(m.group(1))

    scope: Scope = "unknown"
    haystack = (quota_id or "") + " " + text
    if re.search(r"PerDay|per[_ ]?day|daily", haystack, re.I):
        scope = "per_day"
    elif re.search(r"PerMinute|per[_ ]?minute", haystack, re.I):
        scope = "per_minute"

    return RateLimit(scope=scope, retry_after=delay, quota_id=quota_id, quota_value=value)


def is_rate_limit(exc: BaseException) -> bool:
    """Provider-agnostic. langchain_core maps both Google and Anthropic 429s onto
    ModelRateLimitError, so catching that covers a provider switch."""
    try:
        from langchain_core.exceptions import ModelRateLimitError

        if isinstance(exc, ModelRateLimitError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def raise_if_terminal(exc: BaseException) -> None:
    """Convert a daily exhaustion into something a caller will stop on."""
    info = classify(exc)
    if info is not None and not info.retryable:
        raise DailyQuotaExhausted(info.describe()) from exc


def should_retry(exc: BaseException) -> bool:
    """Predicate for LangGraph's RetryPolicy.

    Anything that is not a rate limit retries as normal — a 503 from a busy model
    is exactly what retries are for. A daily cap is the one case where retrying is
    strictly waste.
    """
    info = classify(exc)
    if info is None:
        return True
    return info.retryable
