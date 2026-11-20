"""Tests for the config-driven workflow and provider-agnostic client."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from deepseek_client import LLMClient
from finplan_rag.workflow import load_config
from finplan_rag.workflow import protocol
from run_sequential_evidence import Event, summary_vector
from run_sequential_portfolio import select_policy


def test_protocol_constants():
    assert protocol.TEST_YEARS == (2022, 2023, 2024, 2025)
    assert protocol.FOLDS == 5
    assert protocol.FOLD_SEEDS == (1, 2, 3, 4, 5)
    assert len(protocol.CONFIGS) == 10


def test_config_loads():
    config = load_config("configs/paper.json")
    assert config.data_path.exists()
    assert config.llm["provider"] in {"glm", "deepseek", "openai"}
    assert config.sequential["max_acquire"] >= 1


def test_select_policy_margin():
    losses = {"fixed_adjudicate": 3.0, "planner_evidence": 2.85, "always_wait": 3.2}
    assert select_policy(losses, margin=0.2) == "fixed_adjudicate"
    assert select_policy(losses, margin=0.05) == "planner_evidence"


def test_peer_summary_counts():
    sector = "software"
    events = [
        Event(dt.date(2024, 1, 10), {}, "above", 0.05, sector),
        Event(dt.date(2024, 1, 20), {}, "within", 0.01, "other"),
    ]
    summary = summary_vector(events, dt.date(2024, 1, 15), sector)
    assert summary[0] == 1.0
    assert summary[3] == 1.0  # above rate
    assert summary[5] == 1.0  # same-sector count
    assert summary[6] == 1.0  # same-sector above rate
    later = summary_vector(events, dt.date(2024, 1, 25), sector)
    assert later[0] == 2.0
    assert pytest.approx(later[4]) == 0.03


def test_llm_payload_glm(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    client = LLMClient(
        provider="glm",
        model="glm-5.3-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
    )
    payload = client._payload("hello", "system", 128, 0.0, None, False, 0)
    assert payload["model"] == "glm-5.3-flash"
    assert payload["messages"][0] == {"role": "system", "content": "system"}
    assert payload["messages"][1] == {"role": "user", "content": "hello"}
    assert payload["reasoning_effort"] == "low"


def test_llm_payload_openai_extra_body(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    client = LLMClient(
        provider="openai",
        model="qwen3-8b",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="QWEN_API_KEY",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    payload = client._payload("hello", None, 128, 0.0, None, False, 0)
    assert payload["model"] == "qwen3-8b"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_llm_payload_deepseek(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
    client = LLMClient(
        provider="deepseek",
        model="deepseek-flash[1m]",
        base_url="https://api.deepseek.com/anthropic",
        api_key_env="ANTHROPIC_AUTH_TOKEN",
    )
    payload = client._payload("hello", "system", 128, 0.0, None, False, 0)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["system"] == "system"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_earnings_events_cover_test_claims():
    import json

    path = Path("paper/data/earnings_events.json")
    if not path.exists():
        pytest.skip("earnings events not built")
    events = json.loads(path.read_text(encoding="utf-8"))
    data = json.loads(Path("paper/data/dataset_v4.json").read_text(encoding="utf-8"))
    test = [row for row in data if row.get("split") == "test"]
    assert sum(1 for row in test if row["claim_id"] in events) == len(test)
