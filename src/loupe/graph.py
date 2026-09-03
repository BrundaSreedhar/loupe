"""Graph assembly.

Note what is *not* here: emit. Posting to a PR is a side effect, and keeping it
outside the graph means the eval harness can run thousands of reviews with no
possibility of writing to anyone's repository.
"""

from __future__ import annotations

from collections import defaultdict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from .config import SPECIALIST_ROLES, VERIFY_MODE
from .nodes.dedupe import dedupe
from .nodes.finalize import finalize
from .nodes.prepare import prepare, warm_cache
from .nodes.safety import preflight
from .nodes.specialists import specialist
from .nodes.verify import verify
from .quota import should_retry
from .state import ReviewState

# retry_on excludes a daily quota: three attempts per node against an exhausted
# cap is pure waste, and the server's retryDelay is misleading there — it names a
# per-minute delay even when the window that matters is the day.
_RETRY = RetryPolicy(max_attempts=3, retry_on=should_retry)


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

    by_file: dict[str, list] = defaultdict(list)
    for f in merged:
        by_file[f.file].append(f)

    if VERIFY_MODE == "per_finding":
        # One call per finding: independent judgements, N times the calls and N
        # copies of the same file source.
        return [
            Send("verify", {"path": f.file, "findings": [f], "source": sources.get(f.file, "")})
            for f in merged
        ]

    # One call per file: the source is sent once instead of once per finding, which
    # is both the call-count and the token win. The cost is that the verifier sees
    # every claim on a file at once, so judgements are no longer strictly
    # independent — LOUPE_VERIFY_MODE=per_finding buys that back, and the eval
    # harness can price the difference.
    return [
        Send("verify", {"path": path, "findings": fs, "source": sources.get(path, "")})
        for path, fs in by_file.items()
    ]


def build_graph(checkpointer=None):
    g = StateGraph(ReviewState)

    g.add_node("preflight", preflight)
    g.add_node("prepare", prepare)
    g.add_node("warm_cache", warm_cache, retry_policy=_RETRY)
    g.add_node("specialist", specialist, retry_policy=_RETRY)
    g.add_node("dedupe", dedupe, retry_policy=_RETRY)
    g.add_node("verify", verify, retry_policy=_RETRY)
    g.add_node("finalize", finalize)

    g.add_edge(START, "preflight")
    g.add_edge("preflight", "prepare")
    g.add_edge("prepare", "warm_cache")
    g.add_conditional_edges("warm_cache", fan_out, ["specialist", "dedupe"])
    g.add_edge("specialist", "dedupe")
    g.add_conditional_edges("dedupe", route_verify, ["verify", "finalize"])
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
