"""Phase 2: schema, migrations, conversation persistence and search."""

from __future__ import annotations

import os
import sqlite3
import unittest

from tests.helpers import TMP, fresh_db

import store  # noqa: E402
import vaultdb as v  # noqa: E402


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.path = fresh_db()
        self.s = store.Store(self.path)

    def test_migrations_are_recorded_and_idempotent(self):
        self.assertEqual(self.s.schema_version(), len(store.MIGRATIONS))
        again = store.Store(self.path)
        self.assertEqual(again.schema_version(), len(store.MIGRATIONS))
        tables = {r[0] for r in again.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("users", "conversations", "messages", "memories", "projects",
                  "project_memories", "tasks", "tool_runs", "change_proposals"):
            self.assertIn(t, tables)

    def test_conversation_survives_restart(self):
        cid = self.s.create_conversation("Billing schedule issue", "chat")
        self.s.add_message(cid, "user", "Why are billing schedules not created?")
        self.s.add_message(cid, "assistant", "Check the order product activation.")
        self.s.close()
        reopened = store.Store(self.path)
        msgs = reopened.messages(cid)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(reopened.get_conversation(cid)["title"], "Billing schedule issue")

    def test_rename_tags_category_and_delete_cascade(self):
        cid = self.s.create_conversation("x")
        self.s.add_message(cid, "user", "hello")
        self.s.update_conversation(cid, title="Renamed", category="work", tags=["RCA", "billing", "rca"])
        conv = self.s.get_conversation(cid)
        self.assertEqual(conv["title"], "Renamed")
        self.assertEqual(conv["category"], "work")
        self.assertEqual(conv["tags"], ["billing", "rca"])
        self.assertEqual([c["id"] for c in self.s.list_conversations(tag="rca")], [cid])
        self.assertTrue(self.s.delete_conversation(cid))
        self.assertEqual(self.s.messages(cid), [])
        self.assertEqual(self.s.conn.execute("SELECT COUNT(*) FROM conversation_tags").fetchone()[0], 0)

    def test_full_text_search_finds_message_and_title(self):
        a = self.s.create_conversation("Kafka lag")
        self.s.add_message(a, "user", "Consumer lag alerts fire above ten thousand messages")
        b = self.s.create_conversation("Billing schedules")
        self.s.add_message(b, "user", "Revenue Cloud billing schedule creation fails")
        hits = self.s.search_conversations("billing schedule")
        self.assertEqual(hits[0]["conversation_id"], b)
        self.assertIn("[", hits[0]["snippet"])
        self.assertEqual(self.s.search_conversations("kafka")[0]["conversation_id"], a)
        self.assertEqual(self.s.search_conversations("billing", exclude_id=b), [])

    def test_search_input_cannot_inject_fts_syntax(self):
        cid = self.s.create_conversation("t")
        self.s.add_message(cid, "user", "plain text")
        for nasty in ('"unbalanced', "a OR NOT b*", "col:value", "(((", "NEAR(a b)"):
            self.s.search_conversations(nasty)  # must not raise

    def test_settings_precedence(self):
        self.s.set_setting("TEST_KEY", "saved")
        self.assertEqual(self.s.get_setting("TEST_KEY", "default"), "saved")
        self.assertEqual(self.s.setting_source("TEST_KEY"), "saved")
        os.environ["TEST_KEY"] = "env"
        try:
            self.assertEqual(self.s.get_setting("TEST_KEY", "default"), "env")
            self.assertEqual(self.s.setting_source("TEST_KEY"), "environment")
        finally:
            del os.environ["TEST_KEY"]
        self.assertEqual(self.s.get_setting("MISSING", 3), 3)
        os.environ["NUM_KEY"] = "7"
        try:
            self.assertEqual(self.s.get_setting("NUM_KEY", 1), 7)
        finally:
            del os.environ["NUM_KEY"]

    def test_tasks_and_reminders(self):
        tid = self.s.add_task("Send invoice", due_at=store.now() + 60, remind_at=store.now() - 1)
        self.assertEqual([t["id"] for t in self.s.due_reminders()], [tid])
        self.s.mark_reminded(tid)
        self.assertEqual(self.s.due_reminders(), [])
        self.s.update_task(tid, status="done")
        self.assertEqual(self.s.list_tasks("open"), [])
        self.assertIsNotNone(self.s.get_task(tid)["completed_at"])

    def test_frequent_topics(self):
        for i in range(3):
            cid = self.s.create_conversation(f"c{i}")
            self.s.add_message(cid, "user", "salesforce billing schedule question")
        terms = [t["term"] for t in self.s.frequent_topics(5)]
        self.assertIn("billing", terms)

    def test_legacy_sessions_are_imported_once(self):
        legacy = sqlite3.connect(v.DB_PATH) if v.DB_PATH.parent.exists() else None
        if legacy is None:
            v.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            legacy = sqlite3.connect(v.DB_PATH)
        legacy.executescript(
            "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, title TEXT, profile TEXT,"
            " created_at REAL, updated_at REAL);"
            "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_id INTEGER,"
            " role TEXT, content TEXT, created_at REAL);"
            "DELETE FROM sessions; DELETE FROM messages;"
            "INSERT INTO sessions VALUES (5, 'old chat', 'code', 1, 2);"
            "INSERT INTO messages VALUES (1, 5, 'user', 'old question', 1);"
            "INSERT INTO messages VALUES (2, 5, 'assistant', 'old answer', 1);")
        legacy.commit()
        legacy.close()
        s = store.Store(TMP / "data" / "legacy-import.db")
        convs = s.list_conversations()
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0]["conversation_type"], "coding")
        self.assertEqual([m["content"] for m in s.messages(convs[0]["id"])],
                         ["old question", "old answer"])
        s2 = store.Store(TMP / "data" / "legacy-import.db")
        self.assertEqual(len(s2.list_conversations()), 1)

    def test_split_sql_keeps_triggers_whole(self):
        parts = store._split_sql(store.MIGRATION_1)
        self.assertTrue(all(p.strip().endswith(";") for p in parts))
        self.assertTrue(any("CREATE TRIGGER messages_fts_au" in p and "INSERT INTO messages_fts(rowid"
                            in p for p in parts))


if __name__ == "__main__":
    unittest.main()
