from __future__ import annotations

import importlib


def test_egress_lists_model_and_disabled_tracing(monkeypatch, isolated_config):
    monkeypatch.setenv("LOUPE_MODE", "offline")
    monkeypatch.setenv("LOUPE_PROVIDER", "local")
    config = isolated_config()
    import loupe.privacy as privacy

    importlib.reload(privacy)
    rows = privacy.destinations()
    assert rows[0].destination == config.LOCAL_BASE_URL
    assert rows[1].enabled is False
    assert "loopback" in privacy.assurance()
