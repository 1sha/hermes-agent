from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

ROTATION_DIR = Path.home() / ".hermes" / "claude-max-rotation"
sys.path.insert(0, str(ROTATION_DIR))

import rotation_v2  # noqa: E402


def _base_config():
    return {
        "priority_order": ["spa", "ryan"],
        "accounts": {
            "spa": {"email": "spa@webupon.com"},
            "ryan": {"email": "ryan.lewis@webupon.com"},
        },
        "strike_threshold": 99,
        "strike_window_sec": 300,
        "default_cooldown_sec": 3600,
    }


def _base_state(config=None):
    return rotation_v2._default_state(config or _base_config())


def test_tick_rolls_daily_counters_and_updates_usage_date(monkeypatch):
    config = _base_config()
    state = _base_state(config)
    acct = state["accounts"]["spa"]
    acct["usage_date"] = "2026-04-08"
    acct["requests_today"] = 41
    acct["requests_total"] = 99
    acct["rate_limit_events_today"] = 7
    acct["hard_exhaustions_today"] = 2

    monkeypatch.setattr(rotation_v2, "save_state", lambda _state: None)

    rotation_v2.cmd_tick(state, {**config, "_tick_account_id": "spa"})

    assert acct["usage_date"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert acct["requests_today"] == 1
    assert acct["requests_total"] == 100
    assert acct["rate_limit_events_today"] == 0
    assert acct["hard_exhaustions_today"] == 0


def test_ingest_rate_limit_records_first_quota_exhaustion(monkeypatch):
    config = _base_config()
    state = _base_state(config)
    acct = state["accounts"]["spa"]
    acct["last_activated_at"] = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

    notifications = []
    monkeypatch.setattr(rotation_v2, "save_state", lambda _state: None)
    monkeypatch.setattr(rotation_v2, "_notify_discord", notifications.append)

    result = rotation_v2.ingest_rate_limit(
        state,
        config,
        account_id="spa",
        agent_name="Worker-3a",
        source="run_agent_429_hook",
        status_code=400,
        raw_headers='{"x-test": "1"}',
        error_type="usage_limit_reached",
        message="Out of extra usage until tomorrow",
    )

    assert result is False
    assert acct["usage_date"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert acct["rate_limit_events_today"] == 1
    assert acct["hard_exhaustions_today"] == 1
    assert acct["last_limit_kind"] == "quota_exhausted"
    assert acct["last_limit_source"] == "run_agent_429_hook"
    assert acct["last_limit_status_code"] == 400
    assert acct["first_hard_exhaustion_agent"] == "Worker-3a"
    assert acct["first_hard_exhaustion_source"] == "run_agent_429_hook"
    assert acct["first_hard_exhaustion_status_code"] == 400
    assert acct["first_hard_exhaustion_error_type"] == "usage_limit_reached"
    assert "Out of extra usage" in acct["first_hard_exhaustion_message"]
    assert notifications
    assert any(event["event"] == "hard_exhaustion_recorded" for event in state["recent_events"])


def test_status_displays_daily_usage_and_last_limit_kind(capsys):
    config = _base_config()
    state = _base_state(config)
    acct = state["accounts"]["spa"]
    acct.update({
        "usage_date": "2026-04-09",
        "requests_today": 12,
        "requests_total": 34,
        "rate_limit_events_today": 2,
        "hard_exhaustions_today": 1,
        "last_limit_at": "2026-04-09T19:00:00+00:00",
        "last_limit_source": "run_agent_429_hook",
        "last_limit_kind": "quota_exhausted",
        "last_limit_status_code": 400,
        "first_hard_exhaustion_at": "2026-04-09T19:00:00+00:00",
        "first_hard_exhaustion_agent": "Worker-3a",
    })

    rotation_v2.cmd_status(state, config)
    output = capsys.readouterr().out

    assert "Usage: 2026-04-09 today=12 total=34 rate_limits=2 hard_exhaustions=1" in output
    assert "Last limit: quota_exhausted via run_agent_429_hook [HTTP 400]" in output
    assert "First hard exhaustion: 2026-04-09T19:00:00+00:00 by Worker-3a" in output
