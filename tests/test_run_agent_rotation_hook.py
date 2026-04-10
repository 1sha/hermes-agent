from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def test_try_claude_max_rotation_passes_quota_metadata(agent, monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    rotation_dir = hermes_home / "claude-max-rotation"
    rotation_dir.mkdir(parents=True)
    (rotation_dir / "rotation_v2.py").write_text("# stub\n")
    (hermes_home / ".env").write_text("CLAUDE_CODE_OAUTH_TOKEN=test-token\n")

    monkeypatch.setattr(run_agent.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("CLAUDE_MAX_ACCOUNT_ID", "spa")

    captured = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="rotation triggered", stderr="")

    monkeypatch.setattr(run_agent.subprocess, "run", fake_run)

    result = agent._try_claude_max_rotation(
        reset_time="2026-04-10T00:00:00+00:00",
        retry_after="120",
        status_code=400,
        error_type="usage_limit_reached",
        error_message="Out of extra usage until tomorrow",
    )

    assert result is True
    assert captured["cmd"][:7] == [
        run_agent.sys.executable,
        str(rotation_dir / "rotation_v2.py"),
        "mark-rate-limit",
        "--account",
        "spa",
        "--agent",
        agent.agent_name,
    ]
    assert "--status-code" in captured["cmd"]
    assert "400" in captured["cmd"]
    assert "--error-type" in captured["cmd"]
    assert "usage_limit_reached" in captured["cmd"]
    assert "--message" in captured["cmd"]
    assert "Out of extra usage until tomorrow" in captured["cmd"]
