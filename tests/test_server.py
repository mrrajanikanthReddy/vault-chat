"""HTTP integration: the web API end to end against a fake colibri.

Each test maps to an acceptance criterion: streaming, persistence across a
restart, memory carried into a new conversation and across a model change,
repository-aware coding with diffs applied only on approval, approved tools,
editable memory, metrics, and a working app while colibri is down.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

from tests.helpers import TMP, fresh_db

import chat  # noqa: E402
from tests.fake_colibri import FakeColibri  # noqa: E402


class Client:
    def __init__(self, db=None):
        self.httpd = chat.Server(("127.0.0.1", 0), chat.Handler)
        self.httpd.store_path = db or fresh_db()
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.httpd._api is not None:
            self.httpd._api.p.store.close()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read().decode()
            return json.loads(body) if r.headers.get("Content-Type", "").startswith("application/json") else body

    def post(self, path, body, ctype="application/json"):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": ctype}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def turn(self, **body):
        req = urllib.request.Request(self.base + "/api/turn", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        events = []
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
        return events


def tokens(events):
    return "".join(e["text"] for e in events if e["type"] == "token")


def of(events, kind):
    return [e for e in events if e["type"] == kind]


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeColibri().__enter__()
        self.c = Client()
        self.c.post("/api/settings", {"changes": {"COLIBRI_BASE_URL": self.fake.base_url,
                                                  "USE_NOTES": False}})

    def tearDown(self):
        self.c.close()
        self.fake.__exit__(None, None, None)

    # -- page and transport

    def test_page_is_self_contained(self):
        page = self.c.get("/")
        self.assertIn("<title>Vault Chat</title>", page)
        self.assertIn("<style>", page)
        self.assertNotIn("/*__CSS__*/", page)
        self.assertNotIn("//__JS__", page)
        self.assertNotIn('src="http', page)
        self.assertNotIn("cdn", page.lower())
        for section in ("Chat", "Coding", "Personal Agent", "History", "Memory", "Projects",
                        "Model Settings", "System Health"):
            self.assertIn(f">{section}<", page)

    def test_foreign_host_header_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.c.get("/api/health", {"Host": "attacker.example:80"})
        self.assertEqual(cm.exception.code, 403)

    def test_writes_require_json(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.c.post("/api/memories/create", {"content": "x"}, ctype="text/plain")
        self.assertEqual(cm.exception.code, 415)

    def test_health_shows_provider_and_model(self):
        h = self.c.get("/api/health?force=1")
        self.assertTrue(h["health"]["ok"])
        self.assertEqual(h["provider"], "colibri")
        self.assertEqual(h["model"], "qwen3-coder-30b")
        self.assertEqual(h["context_length"], 16384)

    # -- criteria 2-5: CPU response, streaming, persistence, restart

    def test_streams_and_persists_across_restart(self):
        self.fake.state["reply"] = "Streaming works on the CPU."
        ev = self.c.turn(mode="chat", message="hello there")
        self.assertGreater(len(of(ev, "token")), 3)
        self.assertEqual(tokens(ev), "Streaming works on the CPU.")
        self.assertEqual(ev[-1]["type"], "done")
        stats = of(ev, "stats")[0]
        self.assertIsNotNone(stats["ttft_ms"])
        self.assertGreater(stats["prompt_tokens"], 0)
        cid = of(ev, "conversation")[0]["id"]
        self.assertTrue(self.fake.state["requests"][-1]["stream"])

        db = self.c.httpd.store_path
        self.c.close()
        self.c = Client(db)  # a restart: new server, new platform, same file
        conv = self.c.get(f"/api/conversation?id={cid}")
        self.assertEqual([m["role"] for m in conv["messages"]], ["user", "assistant"])
        self.assertEqual(conv["messages"][1]["content"], "Streaming works on the CPU.")
        self.assertIn(cid, [c["id"] for c in self.c.get("/api/conversations?type=chat")["conversations"]])
        self.assertEqual(self.c.get("/api/conversations?q=hello")["results"][0]["id"], cid)

    # -- criteria 6, 7, 12: memory across conversations and models; editing

    def test_memory_reaches_a_new_conversation_and_survives_a_model_change(self):
        ev = self.c.turn(mode="chat", message="I am working on a Salesforce Revenue Cloud Billing implementation")
        saved = of(ev, "memory")[0]["saved"]
        self.assertTrue(any("Revenue Cloud Billing" in m["content"] for m in saved))

        self.c.turn(mode="chat", message="Help me troubleshoot the Billing Schedule creation issue")
        prompt = json.dumps(self.fake.state["requests"][-1]["messages"])
        self.assertIn("Revenue Cloud Billing", prompt)

        self.fake.state["model"] = "qwen36-35b"   # the server now has another model loaded
        self.c.post("/api/settings", {"changes": {"LLM_MODEL": "qwen36-35b"}})
        ev = self.c.turn(mode="chat", message="What should I check first for billing schedules?")
        self.assertEqual(self.fake.state["requests"][-1]["model"], "qwen36-35b")
        self.assertIn("Revenue Cloud Billing", json.dumps(self.fake.state["requests"][-1]["messages"]))
        self.assertFalse(of(ev, "error"))

    def test_memory_can_be_edited_and_deleted(self):
        m = self.c.post("/api/memories/create", {"content": "I prefer answers with code first",
                                                 "category": "preference"})
        self.c.post("/api/memories/update", {"id": m["id"], "content": "I prefer short answers with code first"})
        items = self.c.get("/api/memories?q=short")["memories"]
        self.assertEqual(items[0]["content"], "I prefer short answers with code first")
        self.c.post("/api/memories/delete", {"id": m["id"]})
        self.assertFalse(self.c.get("/api/memories")["memories"])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.c.post("/api/memories/create", {"content": "my card is 4111 1111 1111 1111"})
        self.assertEqual(cm.exception.code, 400)
        exported = self.c.get("/api/memories/export")
        self.assertIn("memories", exported)

    # -- criteria 8, 9: repository-aware coding, diff before apply

    def test_coding_retrieves_repo_and_applies_only_on_approval(self):
        root = TMP / f"repo-{time.time_ns()}"
        root.mkdir()
        target = root / "billing.py"
        target.write_text("def schedule_count(order):\n    return len(order.items)\n")
        (root / "README.md").write_text("Billing helpers.\n")
        pid = self.c.post("/api/projects/create", {"name": f"billing-{time.time_ns()}",
                                                   "repository_path": str(root)})["id"]
        for _ in range(100):
            if self.c.get(f"/api/project?id={pid}")["index"].get("state") == "done":
                break
            time.sleep(0.05)
        self.fake.state["reply"] = ("Skip empty orders.\n\nbilling.py\n```python\n<<<<<<< SEARCH\n"
                                    "    return len(order.items)\n=======\n"
                                    "    return len([i for i in order.items if i.quantity])\n"
                                    ">>>>>>> REPLACE\n```")
        ev = self.c.turn(mode="coding", project_id=pid, message="schedule_count should skip zero quantity items")
        self.assertIn("schedule_count", json.dumps(self.fake.state["requests"][-1]["messages"]))
        prop = of(ev, "proposal")[0]
        self.assertIn("+    return len([i for i in order.items if i.quantity])", prop["diff"])
        self.assertIn("return len(order.items)", target.read_text())   # untouched until approved
        self.c.post("/api/proposals/apply", {"id": prop["id"]})
        self.assertIn("i.quantity", target.read_text())
        self.c.post("/api/proposals/revert", {"id": prop["id"]})
        self.assertIn("return len(order.items)", target.read_text())

    # -- criteria 10, 11: agent context and approved tools

    def test_agent_tools_wait_for_approval(self):
        root = TMP / f"cmd-{time.time_ns()}"
        root.mkdir()
        pid = self.c.post("/api/projects/create", {"name": f"cmd-{time.time_ns()}",
                                                   "repository_path": str(root)})["id"]
        run = self.c.post("/api/tools/run", {"tool": "run_command", "project_id": pid,
                                             "arguments": {"command": f'{sys.executable} -c "print(7)"'}})
        self.assertEqual(run["status"], "pending_approval")
        pending = self.c.get("/api/tools/runs?status=pending_approval")["runs"]
        self.assertEqual(pending[0]["id"], run["run_id"])
        out = self.c.post("/api/tools/approve", {"run_id": run["run_id"]})
        self.assertEqual(out["status"], "succeeded")
        self.assertEqual(out["result"]["stdout"].strip(), "7")

        ev = self.c.turn(mode="agent", message="Remind me to send the invoice tomorrow at 9am")
        self.assertEqual(of(ev, "action")[0]["tool"], "add_task")
        self.assertEqual(self.c.get("/api/tasks?status=open")["tasks"][0]["title"], "Send the invoice")

    def test_agent_recalls_earlier_conversations(self):
        self.c.turn(mode="chat", message="We decided to generate billing schedules from order activation")
        self.c.turn(mode="agent", message="What did we decide about billing schedules earlier?")
        prompt = json.dumps(self.fake.state["requests"][-1]["messages"])
        self.assertIn("order activation", prompt)
        self.assertIn("<reference", prompt)   # retrieved text is fenced as untrusted

    # -- stop keeps the partial answer

    def test_stop_saves_partial_answer(self):
        self.fake.state["reply"] = " ".join(f"word{i}" for i in range(60))
        self.fake.state["delay"] = 0.05
        body = json.dumps({"mode": "chat", "message": "count slowly"}).encode()
        sock = socket.create_connection(("127.0.0.1", self.c.port), timeout=20)
        sock.sendall(b"POST /api/turn HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        stream = sock.makefile("rb")
        cid, seen = None, 0
        while seen < 3:
            line = stream.readline().decode().strip()
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                cid = ev["id"] if ev["type"] == "conversation" else cid
                seen += ev["type"] == "token"
        stream.close()
        sock.shutdown(socket.SHUT_RDWR)   # what the browser does on Stop
        sock.close()
        for _ in range(100):
            msgs = self.c.get(f"/api/conversation?id={cid}")["messages"]
            if len(msgs) == 2:
                break
            time.sleep(0.05)
        self.assertEqual(len(msgs), 2)
        self.assertTrue(msgs[1]["metadata"].get("stopped"))
        self.assertTrue(msgs[1]["content"].startswith("word0"))

    # -- criterion 13: metrics

    def test_metrics_are_recorded(self):
        self.c.turn(mode="chat", message="measure me")
        m = self.c.get("/api/metrics")
        self.assertGreaterEqual(m["summary"]["n"], 1)
        self.assertIsNotNone(m["recent"][0]["tokens_per_second"])


class ColibriDownTest(unittest.TestCase):
    """Criterion 14: with no inference server the app still works, says why,
    and recovers once the server is back."""

    def setUp(self):
        self.c = Client()
        self.c.post("/api/settings", {"changes": {"COLIBRI_BASE_URL": "http://127.0.0.1:9/v1",
                                                  "USE_NOTES": False}})

    def tearDown(self):
        self.c.close()

    def test_down_then_recovered(self):
        h = self.c.get("/api/health?force=1")
        self.assertFalse(h["health"]["ok"])
        self.assertIn("coli start", h["health"]["hint"])
        ev = self.c.turn(mode="chat", message="are you there?")
        err = of(ev, "error")[0]
        self.assertEqual(err["kind"], "unavailable")
        self.assertTrue(err["retryable"])
        cid = of(ev, "conversation")[0]["id"]
        conv = self.c.get(f"/api/conversation?id={cid}")
        self.assertTrue(conv["messages"][0]["metadata"]["unanswered"])   # the question is kept
        self.assertTrue(self.c.get("/api/conversations")["conversations"])  # history still works
        self.c.post("/api/memories/create", {"content": "I use VS Code", "category": "preference"})

        with FakeColibri() as fake:
            fake.state["reply"] = "Back online."
            self.c.post("/api/settings", {"changes": {"COLIBRI_BASE_URL": fake.base_url}})
            ev = self.c.turn(mode="chat", conversation_id=cid, regenerate=True)
            self.assertEqual(tokens(ev), "Back online.")
            msgs = self.c.get(f"/api/conversation?id={cid}")["messages"]
            self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])

    def test_loading_model_is_reported(self):
        with FakeColibri() as fake:
            fake.state["restarting"] = True
            self.c.post("/api/settings", {"changes": {"COLIBRI_BASE_URL": fake.base_url}})
            h = self.c.get("/api/health?force=1")
            self.assertEqual(h["health"]["state"], "restarting")
            self.assertFalse(h["health"]["ok"])


if __name__ == "__main__":
    unittest.main()
