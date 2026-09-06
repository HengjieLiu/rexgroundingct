# Execution specification

1. Run Exp023 preflight, pinned-manifest download, native audit with lung
   derivation, and deterministic visual QC. Preserve the pending-manual-review
   qualification.
2. Run Exp024 `preflight`, then `prompt-review --split val` and
   `prompt-review --split test`; these manifests are outcome-blind and immutable.
3. Build private test-only native caches for every preprocessing ID required by
   the model manifest. Reuse the sealed iso07 test cache.
4. Run a1 once with Exp007 continuation relative e050/absolute e150; materialize
   a1, a2, and a3 with the fixed 20 mm native-physical-space support.
5. Run each unique oracle checkpoint sequentially on all val200 cases. Compose
   b1 by exact case/finding joins and evaluate b1, exact whole-lung clipping,
   prompt-lung +20 mm, and prompt-fine +20 mm. Write metrics, category/per-case
   tables, TP/FP removal, hit/empty counts, paired CT bootstrap (2,000,
   seed 20260905), and a completion marker. Any unexplained discrepancy stops.
6. Only after the validation marker exists, run the same nine unique checkpoints
   on all 300 test cases and materialize b1, b2, and b3. Reused checkpoint
   outputs are recorded in the manifest.
7. Verify all six directories contain exactly the 300 official filenames and
   native CT-header exports. Do not create a ZIP unless separately requested.

Inference is intentionally delegated to the existing
`run_voxtell_val_inference.py` entrypoint (inside the pinned VoxTell container
when GPU access is required). It now has a test-aware CT-reference exporter;
validation behavior remains unchanged.
