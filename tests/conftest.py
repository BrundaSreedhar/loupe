"""Shared fixtures."""

from __future__ import annotations

import importlib
import sys

import pytest

# Modules that bind values out of config at import time.
_DEPENDANTS = (
    "loupe.nodes.dedupe",
    "loupe.nodes.prepare",
    "loupe.nodes.specialists",
    "loupe.nodes.verify",
    "loupe.nodes.consensus",
    "loupe.nodes.lint",
    "loupe.graph",
)


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Reload `loupe.config` without any .env file influencing the result.

    Config reads three dotenv locations, one of which is the developer's own
    ~/.config/loupe/.env. Without this, a test asserting that a setting is
    unset passes or fails depending on what the person running it happens to have
    configured — which is how installing the tool broke the test suite.

    Returns a callable: set the environment you want, then reload.
    """
    import dotenv

    import loupe.config as config

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    def reload():
        module = importlib.reload(config)
        # Modules that did `from .config import X` hold their own copy of X, so
        # reloading config alone leaves them stale. Reload the dependants too.
        for name in _DEPENDANTS:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
        return module

    yield reload

    # Restore real module state for whatever runs next. Without this a test that
    # reloads config leaks its environment into the rest of the suite — which is
    # how these tests passed alone and failed together.
    monkeypatch.undo()
    reload()
    runner = sys.modules.get("loupe.runner")
    if runner is not None:
        runner._GRAPH = None  # the compiled graph caches the old node wiring
