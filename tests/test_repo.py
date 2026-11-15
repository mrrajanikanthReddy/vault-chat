"""Phase 4: repository indexing, retrieval, edit proposals and approval."""

from __future__ import annotations

import time
import unittest

from tests.helpers import TMP, fresh_db

import repo  # noqa: E402
import store  # noqa: E402

BILLING = '''public with sharing class BillingScheduleService {
    public static void createSchedules(List<OrderItem> items) {
        for (OrderItem item : items) {
            validate(item);
        }
    }

    private static void validate(OrderItem item) {
        if (item.Quantity <= 0) {
            throw new BillingException('Quantity must be positive');
        }
    }
}
'''

UTIL = '''def parse_amount(text):
    """Parse a money amount."""
    return float(text.replace(",", ""))


def format_amount(value):
    return f"{value:,.2f}"
'''


class RepoTest(unittest.TestCase):
    def setUp(self):
        self.s = store.Store(fresh_db())
        self.root = TMP / f"repo-{time.time_ns()}"
        (self.root / "force-app" / "classes").mkdir(parents=True)
        (self.root / "src").mkdir()
        (self.root / "node_modules" / "junk").mkdir(parents=True)
        (self.root / "force-app" / "classes" / "BillingScheduleService.cls").write_text(BILLING)
        (self.root / "src" / "money.py").write_text(UTIL)
        (self.root / "node_modules" / "junk" / "index.js").write_text("module.exports = 1")
        (self.root / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
        (self.root / ".gitignore").write_text("generated/\n")
        (self.root / "generated").mkdir()
        (self.root / "generated" / "out.py").write_text("x = 1\n")
        self.pid = self.s.create_project("Billing", repository_path=str(self.root))

    def test_index_skips_noise_and_is_incremental(self):
        r = repo.index_project(self.s, self.pid)
        self.assertEqual(r["files"], 2)
        paths = {f["path"] for f in repo.list_files(self.s, self.pid)}
        self.assertEqual(paths, {"force-app/classes/BillingScheduleService.cls", "src/money.py"})
        again = repo.index_project(self.s, self.pid)
        self.assertEqual((again["added"], again["updated"], again["unchanged"]), (0, 0, 2))
        (self.root / "src" / "money.py").write_text(UTIL + "\nTAX = 0.2\n")
        (self.root / "force-app" / "classes" / "BillingScheduleService.cls").unlink()
        third = repo.index_project(self.s, self.pid)
        self.assertEqual((third["updated"], third["removed"]), (1, 1))

    def test_symbol_chunks(self):
        chunks = repo.chunk_source(UTIL, "python")
        symbols = {c[2] for c in chunks}
        self.assertIn("parse_amount", symbols)
        apex = repo.chunk_source(BILLING, "apex")
        self.assertEqual(apex[0][2], "BillingScheduleService")

    def test_retrieval_finds_relevant_code(self):
        repo.index_project(self.s, self.pid)
        hits = repo.retrieve(self.s, self.pid, "why does createSchedules throw quantity must be positive?")
        self.assertEqual(hits[0]["path"], "force-app/classes/BillingScheduleService.cls")
        hits = repo.retrieve(self.s, self.pid, "explain money.py")
        self.assertEqual(hits[0]["path"], "src/money.py")

    def test_safe_path_refuses_escapes(self):
        for bad in ("../outside.txt", "/etc/passwd", "src/../../x", ".git/config", ""):
            with self.assertRaises(repo.PathError, msg=bad):
                repo.safe_path(self.root, bad)
        outside = TMP / "outside-target.txt"
        outside.write_text("secret")
        (self.root / "link.txt").symlink_to(outside)
        with self.assertRaises(repo.PathError):
            repo.safe_path(self.root, "link.txt")

    def test_parse_edit_blocks(self):
        text = """Here is the fix:

src/money.py
<<<<<<< SEARCH
    return float(text.replace(",", ""))
=======
    return float(text.replace(",", "").strip())
>>>>>>> REPLACE

```python
src/new_file.py
<<<<<<< SEARCH
=======
VALUE = 1
>>>>>>> REPLACE
```
"""
        edits = repo.parse_edit_blocks(text)
        self.assertEqual([e.path for e in edits], ["src/money.py", "src/new_file.py"])
        self.assertEqual(edits[1].search, "")

    def test_proposal_requires_approval_then_applies_and_reverts(self):
        edits = [repo.Edit("src/money.py", '    return float(text.replace(",", ""))',
                           '    return float(text.replace(",", "").strip())'),
                 repo.Edit("src/new_file.py", "", "VALUE = 1")]
        prop = repo.build_proposal(self.s, self.pid, edits, "strip whitespace")
        self.assertIn('+    return float(text.replace(",", "").strip())', prop["diff"])
        self.assertIn("/dev/null", prop["diff"])
        self.assertEqual((self.root / "src" / "money.py").read_text(), UTIL)  # untouched before approval
        self.assertFalse((self.root / "src" / "new_file.py").exists())
        repo.apply_proposal(self.s, prop["id"])
        self.assertIn(".strip())", (self.root / "src" / "money.py").read_text())
        self.assertEqual((self.root / "src" / "new_file.py").read_text(), "VALUE = 1\n")
        with self.assertRaises(ValueError):
            repo.apply_proposal(self.s, prop["id"])  # not pending any more
        repo.revert_proposal(self.s, prop["id"])
        self.assertEqual((self.root / "src" / "money.py").read_text(), UTIL)
        self.assertFalse((self.root / "src" / "new_file.py").exists())

    def test_stale_file_blocks_apply(self):
        prop = repo.build_proposal(self.s, self.pid, [repo.Edit(
            "src/money.py", "def format_amount(value):", "def format_amount(value, places=2):")])
        (self.root / "src" / "money.py").write_text(UTIL + "# edited meanwhile\n")
        with self.assertRaises(ValueError):
            repo.apply_proposal(self.s, prop["id"])
        self.assertEqual(repo.get_proposal(self.s, prop["id"])["status"], "stale")

    def test_whitespace_tolerant_match_and_errors(self):
        prop = repo.build_proposal(self.s, self.pid, [repo.Edit(
            "src/money.py", "def format_amount(value):\n  return f\"{value:,.2f}\"",
            "def format_amount(value):\n    return f\"{value:,.3f}\"")])
        self.assertIn(",.3f", prop["diff"])
        with self.assertRaises(ValueError):
            repo.build_proposal(self.s, self.pid, [repo.Edit("src/money.py", "no such text", "x")])
        with self.assertRaises(ValueError):
            repo.build_proposal(self.s, self.pid, [repo.Edit("../evil.py", "", "x")])

    def test_reject(self):
        prop = repo.build_proposal(self.s, self.pid, [repo.Edit("src/a.py", "", "A = 1")])
        repo.reject_proposal(self.s, prop["id"])
        self.assertEqual(repo.get_proposal(self.s, prop["id"])["status"], "rejected")
        self.assertFalse((self.root / "src" / "a.py").exists())

    def test_background_index(self):
        self.assertTrue(repo.index_in_background(self.s.path, self.pid))
        for _ in range(100):
            if repo.job_status(self.pid)["state"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(repo.job_status(self.pid)["state"], "done")


if __name__ == "__main__":
    unittest.main()
