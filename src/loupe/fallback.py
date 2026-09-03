"""Falling back to a second model when the day's quota is gone.

LangChain's own `with_fallbacks` takes exception *types*, which is too blunt here:
a per-minute limit and a per-day exhaustion are both 429s, and falling back on the
first would swap models over a seven-second hiccup. Only a confirmed daily
exhaustion is worth switching for; everything else should wait and retry.

The daily cap is per model, so a second model is a second allowance. Falling back
is recorded, not just logged: a review answered partly by another model is a
different artefact, and a set of eval numbers containing both measures neither.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .quota import classify

log = logging.getLogger(__name__)

_state = threading.local()


def reset_usage() -> None:
    _state.used = set()


def record(label: str) -> None:
    if not hasattr(_state, "used"):
        reset_usage()
    _state.used.add(label)


def usage() -> set[str]:
    """Which stages ran on the local model during this thread's review."""
    return set(getattr(_state, "used", set()))


class QuotaFallback:
    """Wraps a primary model call, switching to a spare one on daily exhaustion.

    Only `invoke` is implemented because that is all the nodes call. It is not a
    full Runnable, so it does not appear as its own span in tracing — the inner
    calls still do, which is where the useful detail is anyway.
    """

    def __init__(self, primary: Any, fallback: Any, label: str) -> None:
        self.primary = primary
        self.fallback = fallback
        self.label = label

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return self.primary.invoke(input, config, **kwargs)
        except Exception as exc:
            info = classify(exc)
            if info is None or info.retryable:
                # Not a quota wall — a transient limit, a 503, a bad schema.
                # Let the retry policy deal with it on the good model.
                raise
            log.warning(
                "%s: %s — falling back to the spare model. Findings from this "
                "point come from a different model and are not comparable.",
                self.label, info.describe(),
            )
            record(self.label)
            return self.fallback.invoke(input, config, **kwargs)
