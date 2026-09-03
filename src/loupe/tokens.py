"""What a review cost, in tokens.

Until now a review reported how many findings it produced and how many calls it
took, and nothing about what those calls carried. That makes the two questions
this project keeps asking unanswerable: whether four reviewers earn their keep
over one, and whether the cache warm does what the architecture claims. Both are
about spend, and spend was the one thing not being written down.

Counted with a callback handler passed into the graph config rather than by
threading a return value through every node. The handler is what LangChain hands
usage to when a model replies, so it sees every call — including the ones inside
retries and fallbacks, which a node-level tally would miss.

Kept per model as well as in total, because roles are deliberately pointed at
different models to get separate daily allowances. One combined number would hide
which allowance the spend came out of.
"""

from __future__ import annotations

import threading

from langchain_core.callbacks import UsageMetadataCallbackHandler

from .schema import TokenUsage


class Meter(UsageMetadataCallbackHandler):
    """Token counts plus a call count, which the base handler does not keep.

    The base class locks its own writes because the reviewers run concurrently;
    the call count needs the same protection for the same reason.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self._calls_lock = threading.Lock()

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001 — LangChain's signature
        super().on_llm_end(response, **kwargs)
        with self._calls_lock:
            self.calls += 1


def collect(meter: Meter) -> TokenUsage:
    """Fold what the handler saw into one record."""
    by_model: dict[str, dict[str, int]] = {}
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0}

    for model, usage in (meter.usage_metadata or {}).items():
        inputs = usage.get("input_token_details") or {}
        outputs = usage.get("output_token_details") or {}
        row = {
            "input": usage.get("input_tokens", 0) or 0,
            "output": usage.get("output_tokens", 0) or 0,
            # A cache read is a hit on the prefix the warm wrote; a write is the
            # miss that paid to create it. Reported apart because the whole point
            # of warming is to turn four of the second into one.
            "cache_read": inputs.get("cache_read", 0) or 0,
            # Read from three keys, not one. When Anthropic returns the per-TTL
            # breakdown, langchain_anthropic moves the numbers into the ephemeral
            # keys and sets `cache_creation` to 0 so nothing double-counts — so
            # reading only the obvious key reports every Anthropic cache write as
            # zero, which is exactly the shape of "the warm costs nothing".
            # Verified against the installed package, not recalled.
            "cache_write": (
                (inputs.get("cache_creation") or 0)
                + (inputs.get("ephemeral_5m_input_tokens") or 0)
                + (inputs.get("ephemeral_1h_input_tokens") or 0)
            ),
            "reasoning": outputs.get("reasoning", 0) or 0,
        }
        by_model[model] = row
        for key, value in row.items():
            totals[key] += value

    return TokenUsage(calls=meter.calls, by_model=by_model, **totals)
