"""Targeted routing, gallery, and resumability contracts."""
from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest

import full_gallery as gallery


def route(prompt, fine=False, eligible=True):
    return {"prompt": prompt, "eligible": eligible, "fine_eligible": fine,
            "selected_labels": [10, 11, 12, 13, 14], "scope": "B", "category": "2a"}


class RoutingTests(unittest.TestCase):
    def test_all_three_equivalent_phrasings_enable_upper_lobes(self):
        for prompt in ("Scarring in the upper lobes of both lungs", "Nodules in BOTH UPPER LOBES", "Emphysema in bilateral upper lobes"):
            original = route(prompt)
            before = copy.deepcopy(original)
            effective, corrected = gallery.corrected_route(original)
            self.assertTrue(corrected)
            self.assertTrue(effective["fine_eligible"])
            self.assertEqual(effective["selected_labels"], [10, 12])
            self.assertEqual(original, before)
            self.assertEqual({k for k in effective if effective[k] != original[k]}, {"fine_eligible", "selected_labels"})

    def test_other_routes_and_nonpulmonary_eligibility_are_unchanged(self):
        for original in (route("Opacity in the left lower lobe", fine=True), route("Upper-zone opacity"),
                         route("Bilateral apical changes"), route("Both upper lobes", eligible=False)):
            effective, corrected = gallery.corrected_route(original)
            self.assertEqual(effective, original)
            self.assertFalse(corrected)

    def test_captions_show_correction_support_and_ineligibility(self):
        effective, changed = gallery.corrected_route(route("Both upper lobes"))
        self.assertIn("corrected routing", gallery.captions_for(effective, changed)[2])
        self.assertIn("both upper lobes", gallery.captions_for(effective, changed)[2])
        unmodified = route("Pleural effusion", eligible=False)
        self.assertIn("unchanged", gallery.captions_for(unmodified, False)[1])
        self.assertIn("unchanged", gallery.captions_for(unmodified, False)[2])
        unilateral = route("Right lower lobe opacity", fine=True)
        unilateral["selected_labels"] = [14]
        self.assertIn("right lower lobe", gallery.captions_for(unilateral, False)[2])


class GalleryTests(unittest.TestCase):
    def test_category_page_batches_embeds_and_sorts_findings(self):
        figures = [{"category": "1b", "background": "full_body", "val_index": i // 2,
                    "finding_id": i % 2, "case": f"train_{i // 2}_a_1.nii.gz", "prompt": f"Finding {i}",
                    "path": f"1b/full_body/val{i // 2:03d}_finding{i % 2}.png"} for i in reversed(range(11))]
        counts = {split: Counter({"1b": 11}) for split in ("train", "val", "test")}
        page = gallery.category_page("1b", "full_body", figures, [], counts, dict.fromkeys(counts, 11))
        self.assertEqual(page.count("<details>"), 2)
        self.assertIn("Figures 1–10 of 11", page)
        self.assertIn("Figures 11–11 of 11", page)
        self.assertEqual(page.count("!["), 11)
        self.assertIn("[Other background](lung_only.md)", page)
        self.assertLess(page.index("val000_train_0_a_1 · finding 0"), page.index("val000_train_0_a_1 · finding 1"))

    def test_empty_category_has_two_valid_empty_page_variants(self):
        counts = {s: Counter() for s in ("train", "val", "test")}
        for background in gallery.BACKGROUNDS:
            page = gallery.category_page("2f", background, [], [], counts, dict.fromkeys(counts, 1))
            self.assertIn("zero PNGs", page)
            self.assertNotIn("![", page)
            self.assertNotIn("<details>", page)

    def test_nine_summary_rows_recompose_finding_metrics(self):
        rows = [{"row": role, "policy": policy, "category": "1a", "dice": dice, "hit": dice >= 0.1}
                for role in gallery.pilot.ROW_KEYS for policy in gallery.POLICIES for dice in (0.05, 0.25)]
        summary = gallery.metric_summary(rows, "1a")
        self.assertEqual(len(summary), 9)
        for row in summary:
            self.assertAlmostEqual(row["dice"], 0.15)
            self.assertEqual(row["hits"], 1)
            self.assertEqual(row["findings"], 2)


class ResumeTests(unittest.TestCase):
    def test_resume_requires_matching_settings_complete_census_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"verified artifact")
            item = gallery.file_record(path)
            record = {"status": "COMPLETE", "fingerprint": "configuration A", "routes": {0: {}},
                      "figures": [item, item], "derived_masks": [item] * 4, "metrics": [{}] * 9}
            self.assertTrue(gallery.resume_valid(record, "configuration A"))
            self.assertFalse(gallery.resume_valid(record, "configuration B"))
            incomplete = {**record, "figures": [item]}
            self.assertFalse(gallery.resume_valid(incomplete, "configuration A"))
            path.write_bytes(b"corrupt artifact")
            self.assertFalse(gallery.resume_valid(record, "configuration A"))


if __name__ == "__main__":
    unittest.main()
