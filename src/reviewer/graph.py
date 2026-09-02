"""Graph assembly.

Note what is *not* here: emit. Posting to a PR is a side effect, and keeping it
outside the graph means the eval harness can run thousands of reviews with no
possibility of writing to anyone's repository.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from .config import SPECIALIST_ROLES
from .nodes.dedupe import dedupe
from .nodes.finalize import finalize
from .nodes.prepare import prepare, warm_cache
from .nodes.specialists import specialist
from .nodes.verify import verify
from .state import ReviewState

_RETRY = RetryPolicy(max_attempts=3)


def fan_out(state: ReviewState):
    """Map to reviewers. `single` mode runs one generalist — it is the baseline the
    multi-agent panel has to beat, and it exists so that question stays measurable
    rather than assumed."""
    contexts = state.get("contexts") or {}
    if not contexts:
        return "dedupe"
    roles = SPECIALIST_ROLES if state.get("mode") == "multi" else ("generalist",)
    return [
        Send(
            "specialist",
            {"role": role, "request": state["request"], "contexts": contexts},
        )
        for role in roles
    ]


def route_verify(state: ReviewState):
    merged = state.get("merged") or []
    if not state.get("verify") or not merged:
        return "finalize"

    sources = {
        f.path: f.content_after
        for f in state["request"].files
        if f.content_after is not None
    }
    return [
        Send("verify", {"finding": f, "source": sources.get(f.file, "")})
        for f in merged
    ]


def build_graph(checkpointer=None):
    g = StateGraph(ReviewState)

    g.add_node("prepare", prepare)
    g.add_node("warm_cache", warm_cache, retry_policy=_RETRY)
    g.add_node("specialist", specialist, retry_policy=_RETRY)
    g.add_node("dedupe", dedupe, retry_policy=_RETRY)
    g.add_node("verify", verify, retry_policy=_RETRY)
    g.add_node("finalize", finalize)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "warm_cache")
    g.add_conditional_edges("warm_cache", fan_out, ["specialist", "dedupe"])
    g.add_edge("specialist", "dedupe")
    g.add_conditional_edges("dedupe", route_verify, ["verify", "finalize"])
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
