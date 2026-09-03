"""Console logging.

Every node already logged useful things; none of it was ever configured, so all
of it was discarded. This wires those calls to the terminal and gives the CLI a
verbosity switch.

Third-party loggers are pinned quieter than ours on purpose: at INFO, httpx logs
one line per HTTP request and google.genai logs a retry banner, which buries the
handful of lines that are actually about your review.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_NOISY = ("httpx", "httpcore", "google_genai", "google.genai", "urllib3", "langsmith")


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
    root.setLevel(logging.DEBUG if verbosity >= 2 else logging.INFO)

    logging.getLogger("loupe").setLevel(level)
    logging.getLogger("evals").setLevel(level)

    # At -vv you asked for everything, including the HTTP chatter.
    noisy_level = logging.DEBUG if verbosity >= 3 else logging.WARNING
    for name in _NOISY:
        logging.getLogger(name).setLevel(noisy_level)
