"""Context construction and cache warming."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import specialist_llm
from ..context import build_contexts
from ..prompts.specialists import SHARED_SYSTEM, context_message
from ..state import ReviewState
from ._common import cached_block

log = logging.getLogger(__name__)


def prepare(state: ReviewState, config: RunnableConfig) -> dict:
    contexts, dropped = build_contexts(state["request"])
    return {"contexts": contexts, "dropped": dropped}


def warm_cache(state: ReviewState, config: RunnableConfig) -> dict:
    """Write the shared prefix to cache once, before the fan-out reads it four times.

    Four parallel reviewers all miss a cold cache and all pay to write it (~4x1.25x
    input). Warming first makes it one write plus three reads (~1.25x + 0.3x). The
    cache-hit assertion lives in the specialist node, not here.
    """
    # With one reviewer there is nobody to share the prefix with, so warming is
    # pure added cost. The baseline must not be handicapped by the panel's tax.
    contexts = state.get("contexts") or {}
    if not contexts or state.get("mode") != "multi":
        return {}
    try:
        specialist_llm().bind(max_tokens=16).invoke(
            [
                SystemMessage(SHARED_SYSTEM),
                HumanMessage(content=cached_block(context_message(state["request"], contexts))),
                HumanMessage("Reply with OK."),
            ],
            config={"run_name": "warm_cache", "tags": ["cache"]},
        )
    except Exception as exc:  # noqa: BLE001 — a failed warm is a cost
        # regression, never a correctness one. Never fail a review over it.
        log.warning("cache warm failed (%s); specialists will run cold", exc)
    return {}
