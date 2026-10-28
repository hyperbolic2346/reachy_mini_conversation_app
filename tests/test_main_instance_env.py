"""Tests for CLI instance .env/settings loading path."""

from pathlib import Path

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.main import _load_instance_env_and_settings
from reachy_mini_conversation_app.settings_store import update_settings, AppSettings


def test_cli_loads_default_instance_env_and_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When instance_path is None, the CLI resolves the default path and loads its .env + settings."""
    # Redirect default config home so resolve_instance_path uses our temp location
    monkeypatch.setattr(config_mod, "user_config_dir", lambda name: str(tmp_path / "cfg" / name))
    # Compute default instance folder the helper will resolve to
    default_instance = tmp_path / "cfg" / "reachy_mini" / "conversation_app"
    default_instance.mkdir(parents=True, exist_ok=True)
    # Prepare a .env with an xAI key
    (default_instance / ".env").write_text("XAI_API_KEY=cli-key\n", encoding="utf-8")
    # Prepare settings.json with xAI mode
    update_settings(default_instance, AppSettings(hf_connection_mode="xai"))

    # Call the helper with instance_path=None to simulate CLI launch
    effective_path, _startup = _load_instance_env_and_settings(None, logger=_DummyLogger())

    assert Path(effective_path) == default_instance
    assert config.HF_REALTIME_CONNECTION_MODE == "xai"
    assert config.XAI_API_KEY == "cli-key"


class _DummyLogger:
    """A minimal logger collecting messages without requiring the logging module."""

    def info(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass
