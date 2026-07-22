from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import update_challenge_info as archive  # noqa: E402


class MarkdownConverterTests(unittest.TestCase):
    def test_html_validation_uses_visible_decoded_text(self) -> None:
        result = archive.FetchResult(
            requested_url="https://example.test/",
            final_url="https://example.test/",
            status=200,
            content_type="text/html",
            charset="utf-8",
            etag=None,
            last_modified=None,
            fetched_at_utc="2026-07-21T00:00:00Z",
            content=b"<h1>Register &amp; Submit</h1>",
        )
        archive.validate_fetch(
            {"id": "page", "kind": "html", "validate_contains": ["Register & Submit"]},
            result,
        )

    def test_preserves_structured_content_and_omits_scripts(self) -> None:
        source = b"""
        <html><body>
          <h1>Challenge \xe2\x80\x94 Test</h1>
          <p>Read the <a href="/rules">official rules</a>.</p>
          <ul><li>First</li><li><strong>Second</strong></li></ul>
          <table><tr><th>Model</th><th>Dice</th></tr><tr><td>A|B</td><td>0.500</td></tr></table>
          <pre><code class="language-python">print('ok')</code></pre>
          <form><input name="team" placeholder="Team name"><button>Submit</button></form>
          <a href="javascript:doThing()">Client control</a><button></button>
          <script>FAKE_PRIVATE_DATA</script>
        </body></html>
        """
        output = archive.MarkdownConverter("https://example.test/base/").convert(source)

        self.assertIn("# Challenge \u2014 Test", output)
        self.assertIn("[official rules](https://example.test/rules)", output)
        self.assertIn("- First", output)
        self.assertIn("**Second**", output)
        self.assertIn("| A\\|B | 0.500 |", output)
        self.assertIn("```python", output)
        self.assertIn("[Input: Team name]", output)
        self.assertIn("[Button: Submit]", output)
        self.assertIn("Client control", output)
        self.assertNotIn("javascript:", output)
        self.assertNotIn("[Button: button]", output)
        self.assertNotIn("FAKE_PRIVATE_DATA", output)

    def test_extracts_named_section_only(self) -> None:
        source = b"<html><body><h2>Before</h2><p>x</p><h2>Dataset</h2><p>200 cases</p><h2>After</h2><p>y</p></body></html>"
        output = archive.find_section_markdown(source, "https://example.test/", "Dataset")
        self.assertIn("## Dataset", output)
        self.assertIn("200 cases", output)
        self.assertNotIn("Before", output)
        self.assertNotIn("After", output)

    def test_extracts_section_when_heading_is_wrapped(self) -> None:
        source = b"""
        <html><body><div class="card">
          <div class="headline"><h2>About ReXGroundingCT</h2></div>
          <p>Official dataset description.</p>
          <div class="headline"><h2>Metrics</h2></div>
          <p>Not part of About.</p>
        </div></body></html>
        """
        output = archive.find_section_markdown(source, "https://example.test/", "About ReXGroundingCT")
        self.assertIn("Official dataset description.", output)
        self.assertNotIn("Not part of About.", output)


class FirestoreTests(unittest.TestCase):
    def test_decodes_nested_values(self) -> None:
        payload = {
            "documents": [
                {
                    "name": "projects/p/databases/(default)/documents/leaderboard/a",
                    "fields": {
                        "team": {"stringValue": "A"},
                        "dice": {"doubleValue": 0.5},
                        "totalCases": {"integerValue": "150"},
                        "perCategory": {
                            "mapValue": {
                                "fields": {
                                    "2d": {
                                        "mapValue": {
                                            "fields": {
                                                "n": {"integerValue": "10"},
                                                "dice": {"doubleValue": 0.4},
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    },
                }
            ]
        }
        rows = archive.decode_firestore_documents(payload)
        self.assertEqual(rows[0]["team"], "A")
        self.assertEqual(rows[0]["totalCases"], 150)
        self.assertEqual(rows[0]["perCategory"]["2d"]["n"], 10)

    def test_ranks_best_submission_per_submitter_and_keeps_evaluating(self) -> None:
        rows = [
            {"team": "A2", "submitterKey": "person-a", "dice": 0.8, "status": "published"},
            {"team": "B", "submitterKey": "person-b", "dice": 0.7, "status": "published"},
            {"team": "A1", "submitterKey": "person-a", "dice": 0.6, "status": "published"},
            {"team": "C", "submitterKey": "person-c", "status": "evaluating"},
        ]
        scored, evaluating = archive.rank_challenge_rows(rows)
        self.assertEqual([row["team"] for row in scored], ["A2", "B", "A1"])
        self.assertEqual([row["_archive_rank"] for row in scored], [1, 2, None])
        self.assertEqual([row["_archive_secondary"] for row in scored], [False, False, True])
        self.assertEqual([row["team"] for row in evaluating], ["C"])

    def test_parses_public_firebase_configuration(self) -> None:
        source = b'''<script>var firebaseConfig = {apiKey: "public-key", projectId: "public-project"};</script>'''
        self.assertEqual(archive.parse_firebase_config(source), ("public-project", "public-key"))


class IntegrityTests(unittest.TestCase):
    def test_verifies_manifest_hash_and_detects_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            raw = snapshot / "raw" / "source.html"
            raw.parent.mkdir()
            raw.write_bytes(b"official")
            generated = snapshot / "generated" / "page.md"
            generated.parent.mkdir()
            generated.write_bytes(b"# Page\n")
            manifest = {
                "sources": [
                    {
                        "id": "source",
                        "snapshot_path": "raw/source.html",
                        "bytes": len(b"official"),
                        "sha256": archive.sha256_bytes(b"official"),
                    }
                ],
                "generated": [
                    {
                        "path": "generated/page.md",
                        "bytes": len(b"# Page\n"),
                        "sha256": archive.sha256_bytes(b"# Page\n"),
                    }
                ],
            }
            (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(archive.verify_snapshot(snapshot), [])
            raw.write_bytes(b"changed")
            self.assertTrue(any("SHA-256 mismatch" in error for error in archive.verify_snapshot(snapshot)))

    def test_promotion_rolls_back_current_documents_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            generated = root / "generated"
            generated.mkdir()
            (root / "a.md").write_text("old a", encoding="utf-8")
            (root / "b.md").write_text("old b", encoding="utf-8")
            (generated / "a.md").write_text("new a", encoding="utf-8")
            (generated / "b.md").write_text("new b", encoding="utf-8")

            real_replace = os.replace
            failed = False

            def replace_once_then_work(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None:
                nonlocal failed
                destination_path = Path(destination)
                if destination_path == root / "b.md" and not failed:
                    failed = True
                    raise OSError("simulated promotion failure")
                real_replace(source, destination)

            with mock.patch.object(archive.os, "replace", side_effect=replace_once_then_work):
                with self.assertRaises(OSError):
                    archive.promote_documents(generated, root, ["a.md", "b.md"])

            self.assertEqual((root / "a.md").read_text(encoding="utf-8"), "old a")
            self.assertEqual((root / "b.md").read_text(encoding="utf-8"), "old b")


if __name__ == "__main__":
    unittest.main()
