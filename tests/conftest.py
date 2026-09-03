"""Shared fixtures."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Reload `reviewer.config` without any .env file influencing the result.

    Config reads three dotenv locations, one of which is the developer's own
    ~/.config/reviewer/.env. Without this, a test asserting that a setting is
    unset passes or fails depending on what the person running it happens to have
    configured — which is how installing the tool broke the test suite.

    Returns a callable: set the environment you want, then reload.
    """
    import dotenv

    import reviewer.config as config

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    def reload():
        return importlib.reload(config)

    yield reload
    # Restore the real module state for anything that runs after this test.
    monkeypatch.undo()
    importlib.reload(config)
