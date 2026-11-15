"""Phase 5: tools, approvals, intent detection and full orchestrated turns."""

from __future__ import annotations

import datetime as dt
import sys
import time
import unittest

from tests.helpers import TMP, fresh_db

import orchestrator as orch  # noqa: E402
import repo  # noqa: E402
from tests.fake_colibri import FakeColibri  # noqa: E402
from tools import ToolContext, ToolError, check_command, parse_when  # noqa: E402


class WhenTest(unittest.TestCase):
    now = dt.datetime(2026, 11, 15, 12, 0)  # a Sunday

    def at(self, text):
        ts = parse_when(text, self.now)
        return dt.datetime.fromtimestamp(ts) if ts else None

    def test_phrases(self):
        self.assertEqual(self.at("in 2 hours"), dt.datetime(2026, 11, 15, 14, 0))
        self.assertEqual(self.at("in 30 minutes"), dt.datetime(2026, 11, 15, 12, 30))
        self.assertEqual(self.at("tomorrow at 9am"), dt.datetime(2026, 11, 16, 9, 0))
        self.assertEqual(self.at("tomorrow"), dt.datetime(2026, 11, 16, 9, 0))
        self.assertEqual(self.at("today at 5pm"), dt.datetime(2026, 11, 15, 17, 0))
        self.assertEqual(self.at("tonight"), dt.datetime(2026, 11, 15, 20, 0))
        self.assertEqual(self.at("next friday"), dt.datetime(2026, 11, 20, 9, 0))
        self.assertEqual(self.at("at 10:30"), dt.datetime(2026, 11, 16, 10, 30))
        self.assertEqual(self.at("2026-12-01 14:00"), dt.datetime(2026, 12, 1, 14, 0))
        self.assertIsNone(self.at("whenever"))


class IntentTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(orch.detect_intent("remind me to call Bob tomorrow", "chat").kind, "action")
        self.assertEqual(orch.detect_intent("What project am I currently working on?", "agent").kind, "recall")
        self.assertEqual(orch.detect_intent("What did we discuss about my previous implementation?",
                                            "agent").kind, "recall")
        self.assertEqual(orch.detect_intent("Prepare a technical implementation plan", "agent").kind, "planning")
        self.assertEqual(orch.detect_intent("Traceback (most recent call last): KeyError", "chat").kind, "coding")
        self.assertEqual(orch.detect_intent("how was your day", "chat").kind, "conversational")

    def test_plan_actions(self):
        self.assertEqual(orch.plan_actions("Remind me to send the invoice tomorrow at 9am", "chat"),
                         [("add_task", {"title": "Send the invoice", "due": "tomorrow at 9am",
                                        "remind": "tomorrow at 9am"})])
        self.assertEqual(orch.plan_actions("add a task: review PR 42", "agent"),
                         [("add_task", {"title": "Review PR 42"})])
        self.assertEqual(orch.plan_actions("show my tasks", "agent"), [("list_tasks", {"status": "open"})])
        self.assertEqual(orch.plan_actions("mark task 3 done", "agent"), [("complete_task", {"task_id": 3})])
        self.assertEqual(orch.plan_actions("run `pytest -q`", "coding"), [("run_command", {"command": "pytest -q"})])


class ToolRegistryTest(unittest.TestCase):
    def setUp(self):
        self.p = orch.Platform(fresh_db())
        self.root = TMP / f"cmdrepo-{time.time_ns()}"
        self.root.mkdir()
        pid = self.p.store.create_project("cmd", repository_path=str(self.root))
        self.ctx = ToolContext(self.p.store, self.p.memory, self.p.store.get_project(pid))

    def test_allowlist_is_enforced_and_logged(self):
        res = self.p.registry.invoke("run_command", {"command": "ls"}, self.ctx, allowlist=["list_tasks"])
        self.assertEqual(res["status"], "failed")
        runs = self.p.registry.runs(self.p.store)
        self.assertEqual(runs[0]["status"], "failed")

    def test_command_needs_approval_then_runs(self):
        res = self.p.registry.invoke("run_command", {"command": f'{sys.executable} -c "print(42)"'}, self.ctx)
        self.assertEqual(res["status"], "pending_approval")
        self.assertEqual(self.p.registry.runs(self.p.store, "pending_approval")[0]["id"], res["run_id"])
        out = orch.approve_tool(self.p, res["run_id"])
        self.assertEqual(out["status"], "succeeded")
        self.assertEqual(out["result"]["stdout"].strip(), "42")
        with self.assertRaises(ToolError):
            orch.approve_tool(self.p, res["run_id"])

    def test_unquoted_program_path_with_spaces(self):
        folder = TMP / f"dir with space {time.time_ns()}"
        folder.mkdir()
        exe = folder / "python"
        exe.symlink_to(sys.executable)
        res = self.p.registry.invoke("run_command", {"command": f'{exe} -c "print(42)"'}, self.ctx)
        out = orch.approve_tool(self.p, res["run_id"])
        self.assertEqual(out["status"], "succeeded", out.get("error"))
        self.assertEqual(out["result"]["stdout"].strip(), "42")

    def test_reject(self):
        res = self.p.registry.invoke("run_command", {"command": "ls"}, self.ctx)
        self.p.registry.reject(res["run_id"], self.p.store)
        self.assertEqual(self.p.registry.runs(self.p.store)[0]["status"], "rejected")

    def test_dangerous_commands(self):
        self.assertIsNotNone(check_command("rm -rf build"))
        self.assertIsNotNone(check_command("git push --force origin main"))
        self.assertIsNone(check_command("pytest -q"))
        for cmd in ("ls | wc -l", "echo hi > f.txt", "sudo ls"):
            res = self.p.registry.invoke("run_command", {"command": cmd}, self.ctx)
            out = orch.approve_tool(self.p, res["run_id"])
            self.assertEqual(out["status"], "failed", cmd)

    def test_missing_arguments(self):
        res = self.p.registry.invoke("add_task", {}, self.ctx)
        self.assertEqual(res["status"], "failed")
        self.assertIn("title", res["error"])

    def test_untrusted_reads_withdraw_file_tools(self):
        names = ["search_code", "read_file", "run_command"]
        all_tools = {t["function"]["name"] for t in self.p.registry.schemas(names, project=self.ctx.project)}
        safe = {t["function"]["name"] for t in self.p.registry.schemas(names, True, self.ctx.project)}
        self.assertEqual(all_tools, set(names))
        self.assertEqual(safe, {"search_code"})


def run(orchestrator, **req):
    return list(orchestrator.run_turn(req))


def kinds(events):
    return [e["type"] for e in events]


class OrchestratorTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeColibri().__enter__()
        self.p = orch.Platform(fresh_db())
        self.p.update_settings({"COLIBRI_BASE_URL": self.fake.base_url, "USE_NOTES": False})
        self.o = orch.Orchestrator(self.p)

    def tearDown(self):
        self.fake.__exit__()

    def test_chat_turn_streams_and_persists(self):
        self.fake.state["reply"] = "Billing schedules come from order products."
        ev = run(self.o, mode="chat", message="How are billing schedules created?")
        self.assertEqual(kinds(ev)[0], "conversation")
        self.assertIn("token", kinds(ev))
        self.assertEqual(kinds(ev)[-1], "done")
        cid = ev[0]["id"]
        msgs = self.p.store.messages(cid)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["content"], "Billing schedules come from order products.")
        self.assertEqual(msgs[1]["metadata"]["provider"], "colibri")
        self.assertEqual(self.p.store.recent_metrics()[0]["ok"], 1)
        sent = self.fake.state["requests"][-1]
        self.assertEqual(sent["temperature"], 0.2)
        self.assertEqual(sent["max_tokens"], 1024)

    def test_memory_carries_into_a_new_conversation(self):
        run(self.o, mode="chat", message="I am working on a Salesforce Revenue Cloud Billing implementation.")
        ev = run(self.o, mode="chat", message="Help me troubleshoot the Billing Schedule creation issue.")
        last_user = self.fake.state["requests"][-1]["messages"][-1]["content"]
        self.assertIn("Revenue Cloud Billing", last_user)
        context = next(e for e in ev if e["type"] == "context")
        self.assertTrue(any(src["kind"] == "memory" for src in context["sources"]))

    def test_follow_up_sends_history(self):
        ev = run(self.o, mode="chat", message="first question")
        run(self.o, mode="chat", message="second question", conversation_id=ev[0]["id"])
        roles = [m["role"] for m in self.fake.state["requests"][-1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_regenerate_replaces_last_answer(self):
        ev = run(self.o, mode="chat", message="tell me a fact")
        cid = ev[0]["id"]
        self.fake.state["reply"] = "A different answer."
        run(self.o, mode="chat", conversation_id=cid, regenerate=True)
        msgs = self.p.store.messages(cid)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["content"], "A different answer.")

    def test_stop_keeps_partial_answer(self):
        self.fake.state["reply"] = " ".join(f"w{i}" for i in range(300))
        self.fake.state["delay"] = 0.005
        gen = self.o.run_turn({"mode": "chat", "message": "write a lot"})
        cid, tokens = None, 0
        for e in gen:
            if e["type"] == "conversation":
                cid = e["id"]
            if e["type"] == "token":
                tokens += 1
                if tokens == 5:
                    break
        gen.close()
        msgs = self.p.store.messages(cid)
        self.assertEqual(msgs[-1]["role"], "assistant")
        self.assertTrue(msgs[-1]["metadata"].get("stopped"))
        self.assertTrue(msgs[-1]["content"].startswith("w0 w1"))

    def test_provider_down_is_reported_and_history_kept(self):
        self.p.update_settings({"COLIBRI_BASE_URL": "http://127.0.0.1:9/v1"})
        ev = run(self.o, mode="chat", message="are you there?")
        err = next(e for e in ev if e["type"] == "error")
        self.assertEqual(err["kind"], "unavailable")
        self.assertTrue(err["retryable"])
        self.assertIn("coli start", err["hint"])
        msgs = self.p.store.messages(ev[0]["id"])
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0]["metadata"]["unanswered"])
        self.p.update_settings({"COLIBRI_BASE_URL": self.fake.base_url})
        again = run(self.o, mode="chat", conversation_id=ev[0]["id"], regenerate=True)
        self.assertEqual(kinds(again)[-1], "done")
        self.assertEqual(len(self.p.store.messages(ev[0]["id"])), 2)

    def test_reminder_is_created_without_a_model_tool_call(self):
        ev = run(self.o, mode="agent", message="Remind me to send the invoice tomorrow at 9am")
        action = next(e for e in ev if e["type"] == "action")
        self.assertEqual(action["tool"], "add_task")
        tasks = self.p.store.list_tasks()
        self.assertEqual(tasks[0]["title"], "Send the invoice")
        self.assertIsNotNone(tasks[0]["remind_at"])
        self.assertIn("Task #", self.fake.state["requests"][-1]["messages"][-1]["content"])

    def test_agent_tool_loop(self):
        self.p.store.add_task("Prepare demo")
        self.fake.state["tool_calls"] = [[{"id": "c1", "type": "function", "function": {
            "name": "list_tasks", "arguments": "{}"}}]]
        ev = run(self.o, mode="agent", message="could you schedule my day based on my todo items")
        action = next(e for e in ev if e["type"] == "action")
        self.assertEqual(action["tool"], "list_tasks")
        self.assertEqual(action["status"], "succeeded")
        self.assertIn("Prepare demo", self.fake.state["requests"][-1]["messages"][-1]["content"])

    def test_agent_recall_uses_previous_conversations(self):
        run(self.o, mode="chat", message="We decided to use Queueable Apex for billing schedule retries.")
        run(self.o, mode="agent", message="What did we discuss about billing schedule retries?")
        last_user = self.fake.state["requests"][-1]["messages"][-1]["content"]
        self.assertIn("Queueable Apex", last_user)
        self.assertIn("earlier conversations", last_user)

    def test_coding_turn_produces_a_diff_proposal(self):
        root = TMP / f"coderepo-{time.time_ns()}"
        root.mkdir()
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        pid = self.p.store.create_project("calc", repository_path=str(root))
        repo.index_project(self.p.store, pid)
        self.fake.state["reply"] = ("The sign is wrong.\n\ncalc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n"
                                    "    return a + b\n>>>>>>> REPLACE\n")
        ev = run(self.o, mode="coding", message="Why does add() in calc.py subtract?", project_id=pid)
        prompt = self.fake.state["requests"][-1]["messages"][-1]["content"]
        self.assertIn("return a - b", prompt)
        prop = next(e for e in ev if e["type"] == "proposal")
        self.assertIn("+    return a + b", prop["diff"])
        self.assertEqual((root / "calc.py").read_text(), "def add(a, b):\n    return a - b\n")
        repo.apply_proposal(self.p.store, prop["id"])
        self.assertIn("a + b", (root / "calc.py").read_text())

    def test_settings_change_switches_provider(self):
        first = self.p.provider()
        self.p.update_settings({"LLM_MODEL": "qwen3-coder-30b"})
        self.assertIsNot(first, self.p.provider())
        self.assertEqual(self.p.provider().active_model(), "qwen3-coder-30b")
        with self.assertRaises(ValueError):
            self.p.update_settings({"NOT_A_SETTING": 1})


if __name__ == "__main__":
    unittest.main()
