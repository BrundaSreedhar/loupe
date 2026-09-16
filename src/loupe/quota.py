"""Which failures are worth continuing past, and which are not.

Mostly this is about telling the two kinds of 429 apart.

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


class AuthenticationFailed(RuntimeError):
    """The credential was rejected. Every later call fails identically."""


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


def is_auth_error(exc: BaseException) -> bool:
    """A rejected credential, whichever provider rejected it.

    Distinct from a missing one, which `config.require_credentials` catches before
    a run starts. This is the key that is present, well-formed, and not accepted —
    which cannot be known without asking, and once asked is true for every call
    that follows.
    """
    if getattr(exc, "status_code", None) == 401 or getattr(exc, "code", None) == 401:
        return True
    text = str(exc)
    return bool(
        re.search(
            r"authentication_error|invalid_api_key|API[_ ]KEY[_ ]INVALID"
            r"|API key is invalid|UNAUTHENTICATED",
            text,
            re.I,
        )
    )


def raise_if_terminal(exc: BaseException) -> None:
    """Convert a failure that cannot improve into something a caller will stop on.

    Called first inside a broad except, so that a handler written to absorb a bad
    response does not also absorb the reason every later response will be bad.
    """
    if is_auth_error(exc):
        raise AuthenticationFailed(
            "the provider rejected the credential — every call in this run will "
            "fail the same way"
        ) from exc
    info = classify(exc)
    if info is not None and not info.retryable:
        raise DailyQuotaExhausted(info.describe()) from exc


def should_retry(exc: BaseException) -> bool:
    """Predicate for LangGraph's RetryPolicy.

    Anything that is not a rate limit retries as normal — a 503 from a busy model
    is exactly what retries are for. A daily cap is the one case where retrying is
    strictly waste.
    """
    # The translated exceptions first, and by type. Once a node has caught a raw
    # provider error and re-raised it as one of these, the provider's own wording
    # is gone — and these classifiers read wording. Matching only the raw shape
    # meant a node that correctly identified a terminal failure had its verdict
    # thrown away by the retry policy, which then ran it twice more.
    if isinstance(exc, AuthenticationFailed | DailyQuotaExhausted):
        return False
    if is_auth_error(exc):
        return False
    info = classify(exc)
    if info is None:
        return True
    return info.retryable
