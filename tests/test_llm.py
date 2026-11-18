"""Offline checks for the LLM client: the Claude backend parses replies, records cost, caches, and
stops at the spend cap; open-weight cache keys are unchanged by the Claude support."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from mastermind import llm


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="check clue 1"),
                     SimpleNamespace(type="text", text='{"guess": "1234"}')],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500),
            stop_reason="end_turn", model="claude-opus-5-5")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("SPEND_CAP_USD", raising=False)
    c = llm.Client("claude-opus-5-5")
    c.ledger = tmp_path / c.key / "_spend.jsonl"
    c.spent = 0.0
    c._client = SimpleNamespace(messages=FakeMessages())
    return c


def test_claude_request_and_record(client):
    rec = client.complete("prompt", sample=0)
    kw = client._client.messages.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 8192
    assert kw["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k", "fallbacks"} & set(kw)  # rejected or unwanted
    assert rec["reply"] == '{"guess": "1234"}' and rec["reasoning"] == "check clue 1"
    assert rec["cost_usd"] == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    assert client.complete("prompt", sample=0)["cached"] and len(client._client.messages.calls) == 1


def test_spend_cap(client):
    client.cap = 0.025  # each fake call costs $0.014: two calls allowed, the third refused
    client.complete("a")
    client.complete("b")
    with pytest.raises(llm.SpendCapReached):
        client.complete("c")
    assert client._ledger_total() == pytest.approx(0.028)


def test_open_weight_cache_key_unchanged():
    c = llm.Client("qwen3-32b")
    reg = llm.load_registry()["qwen3-32b"]
    settings = {"model": reg["hf_id"], "revision": reg["revision"], **reg["decoding"], "extra": reg["thinking_off"]}
    key = hashlib.sha256(json.dumps([settings, "p", 0], sort_keys=True).encode()).hexdigest()
    assert c._cache_path("p", 0).name == f"{key}.json"
