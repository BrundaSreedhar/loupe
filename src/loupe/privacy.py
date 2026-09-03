"""Explicitly describe where a review may send data.

This is intentionally a small, dependency-free module: the privacy contract
must remain inspectable even when an LLM SDK is misconfigured or unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import LOCAL_BASE_URL, MODE, PROVIDER, TRACING


@dataclass(frozen=True)
class EgressDestination:
    destination: str
    data: str
    enabled: bool


def destinations(include_github: bool = False) -> list[EgressDestination]:
    """The complete list of network destinations Loupe itself can use.

    ``include_github`` is false for the generic doctor command because a local
    review does not contact GitHub; the PR command opts in to that row.
    """
    local = PROVIDER == "local"
    endpoint = LOCAL_BASE_URL if local else PROVIDER
    return [
        EgressDestination(
            endpoint,
            "review prompts: changed source, selected repository context, project rules, "
            "and findings",
            True,
        ),
        EgressDestination(
            "LangSmith",
            "trace metadata and, depending on LangChain instrumentation, prompt and response "
            "content",
            TRACING,
        ),
        EgressDestination(
            "GitHub API",
            "PR metadata and source retrieval; review comments only when --post is confirmed",
            include_github,
        ),
    ]


def assurance() -> str:
    if MODE == "offline":
        return (
            "offline: only a loopback local model endpoint is permitted; tracing and PR access "
            "are disabled"
        )
    if MODE == "private":
        return (
            "private: tracing is disabled; model traffic is sent only to the provider endpoint "
            "you configured"
        )
    return (
        "cloud: model traffic is sent to the configured provider; tracing requires "
        "LOUPE_TRACING=on"
    )
