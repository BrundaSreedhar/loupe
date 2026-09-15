"""Console logging.

Every node already logged useful things; none of it was ever configured, so all
of it was discarded. This wires those calls to the terminal and gives the CLI a
verbosity switch.

Third-party loggers are pinned quieter than ours on purpose: at INFO, httpx logs
one line per HTTP request and google.genai logs a retry banner, which buries the
handful of lines that are actually about your review.

The root logger is the default-deny half of that. Naming noisy libraries one by
one only silences the ones already known: the Anthropic SDK moved to `httpx2`,
which was not on the list, so every model call printed a request line straight
through a running spinner. Root sits at WARNING and the two loggers we own are
raised explicitly, so a new dependency is quiet until someone decides otherwise.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

# httpx2/httpcore2 are what the Anthropic SDK talks through, httpx/httpcore what
# google-genai and the GitHub adapter use. Which pair is loud depends on the
# provider, so both are named.
_NOISY = (
    "httpx",
    "httpcore",
    "httpx2",
    "httpcore2",
    "google_genai",
    "google.genai",
    "urllib3",
    "langsmith",
)

# Warnings from a dependency about how *another* dependency calls it. Nothing the
# user of this tool can act on, and they fire on every single model call. Dropped
# by message rather than by silencing the whole logger, so a genuine SDK warning
# still gets through.
_UNACTIONABLE = (
    "automatic function calling",
    "AFC is enabled",
    "AFC remote call",
)


class _DropUnactionable(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in _UNACTIONABLE)


def setup_logging(verbosity: int = 0, console: Console | None = None) -> None:
    """verbosity 0 = warnings only, 1 = what each stage did, 2 = everything."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)

    handler = RichHandler(
        console=console or Console(stderr=True),
        show_path=False,
        show_time=verbosity >= 2,
        rich_tracebacks=True,
        markup=False,
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    # WARNING, not INFO. A library inherits this unless it is named below, so an
    # unlisted dependency is quiet by default rather than loud by default. The two
    # loggers we own are raised straight after, and a record that passes its own
    # logger's level still reaches the handler whatever root's level is.
    root.setLevel(logging.DEBUG if verbosity >= 2 else logging.WARNING)

    logging.getLogger("loupe").setLevel(level)
    logging.getLogger("evals").setLevel(level)

    # At -vv you asked for everything, including the HTTP chatter.
    noisy_level = logging.DEBUG if verbosity >= 3 else logging.WARNING
    for name in _NOISY:
        logging.getLogger(name).setLevel(noisy_level)

    # At -vvv you asked for everything, warts included.
    if verbosity < 3:
        handler.addFilter(_DropUnactionable())
