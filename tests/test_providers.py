"""Phase 1: the inference layer against a fake colibri gateway."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers import (ColibriProvider, GenParams, ProviderError, make_provider)  # noqa: E402
from tests.fake_colibri import FakeColibri  # noqa: E402


class ColibriProviderTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeColibri().__enter__()
        self.p = ColibriProvider(self.fake.base_url, retries=1)

    def tearDown(self):
        self.fake.__exit__()

    def test_health_reports_model_and_context(self):
        h = self.p.health(force=True)
        self.assertTrue(h["ok"])
        self.assertEqual(h["state"], "ready")
        self.assertEqual(h["model"], "qwen3-coder-30b")
        self.assertEqual(h["context_length"], 16384)

    def test_health_while_restarting(self):
        self.fake.state["restarting"] = True
        h = self.p.health(force=True)
        self.assertFalse(h["ok"])
        self.assertEqual(h["state"], "restarting")

    def test_health_when_down_gives_hint(self):
        p = ColibriProvider("http://127.0.0.1:9/v1")
        h = p.health(force=True)
        self.assertFalse(h["ok"])
        self.assertEqual(h["state"], "down")
        self.assertIn("coli start", h["hint"])

    def test_streams_tokens_and_collects_stats(self):
        self.fake.state["reply"] = "one two three four"
        stream = self.p.stream_chat([{"role": "user", "content": "hi"}],
                                    GenParams(temperature=0.3, top_p=0.9, max_tokens=64))
        pieces = list(stream)
        self.assertEqual("".join(pieces), "one two three four")
        self.assertEqual(len(pieces), 4)
        s = stream.stats.as_dict()
        self.assertEqual(s["completion_tokens"], 4)
        self.assertIsNotNone(s["ttft_ms"])
        self.assertEqual(s["queue_wait_ms"], 3.0)
        sent = self.fake.state["requests"][-1]
        self.assertEqual(sent["temperature"], 0.3)
        self.assertEqual(sent["top_p"], 0.9)
        self.assertEqual(sent["max_tokens"], 64)
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["context_compaction"], "auto")
        self.assertNotIn("cache_slot", sent)

    def test_kv_slot_per_conversation(self):
        p = ColibriProvider(self.fake.base_url, kv_slots=4)
        list(p.stream_chat([{"role": "user", "content": "x"}], conversation_id=7))
        self.assertEqual(self.fake.state["requests"][-1]["cache_slot"], 3)

    def test_retries_a_loading_engine(self):
        self.fake.state["fail_next"] = [503]
        self.assertEqual(self.p.chat([{"role": "user", "content": "hi"}]), "Hello from colibri.")

    def test_full_queue_is_classified(self):
        self.fake.state["fail_next"] = [429, 429]
        with self.assertRaises(ProviderError) as ctx:
            list(self.p.stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(ctx.exception.kind, "busy")
        self.assertTrue(ctx.exception.retryable)

    def test_context_overflow_is_classified(self):
        self.fake.state["fail_next"] = [400]
        with self.assertRaises(ProviderError) as ctx:
            self.p.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, "context")

    def test_unknown_model_is_classified(self):
        p = ColibriProvider(self.fake.base_url, model="not-loaded")
        with self.assertRaises(ProviderError) as ctx:
            p.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, "model")

    def test_unreachable_server(self):
        p = ColibriProvider("http://127.0.0.1:9/v1")
        with self.assertRaises(ProviderError) as ctx:
            p.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, "unavailable")

    def test_complete_returns_tool_calls(self):
        self.fake.state["tool_calls"] = [[{"id": "c1", "type": "function", "function": {
            "name": "list_tasks", "arguments": "{}"}}]]
        out = self.p.complete([{"role": "user", "content": "tasks?"}],
                              tools=[{"type": "function", "function": {"name": "list_tasks"}}])
        self.assertEqual(out["tool_calls"][0]["function"]["name"], "list_tasks")
        self.assertIn("tools", self.fake.state["requests"][-1])

    def test_close_cancels_generation(self):
        self.fake.state["reply"] = " ".join(["w"] * 200)
        self.fake.state["delay"] = 0.01
        stream = self.p.stream_chat([{"role": "user", "content": "long"}])
        it = iter(stream)
        next(it)
        stream.close()
        self.assertLess(stream.stats.pieces, 200)

    def test_factory_defaults_to_colibri_url(self):
        p = make_provider("colibri")
        self.assertEqual(p.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(p.root, "http://127.0.0.1:8000")
        self.assertEqual(make_provider("ollama").base_url, "http://127.0.0.1:11434/v1")
        with self.assertRaises(ValueError):
            make_provider("nope")


if __name__ == "__main__":
    unittest.main()
