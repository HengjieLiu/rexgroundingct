# Frozen collaborator source

These Python files are byte-for-byte copies of the dependency closure of the
two deployment scripts from the user-provided collaborator archive.
`manifest.json` records archive and member hashes and original paths. Preserve
them unchanged; corrections or retuning require a new method identity.

The source is attributed to the collaborator's supplied
RexGrounding_challenge2026-c2-less-semantic-coadapt-v11-freeze bundle. The ZIP
contains no root license file; no additional license is asserted here.

`semantic_v1_policy.json` is a reconstructed configuration, not a file found in
the ZIP. Its groups reproduce the assertions in `load_config` of
`apply_semantic_constraint_best_policy.py`; the referenced outputs YAML is
excluded from the archive. It is JSON, which the original YAML loader accepts.

No training, TotalSegmentator execution or original audit main entrypoint is
launched. Production imports parser/proxy functions; focused tests compare the
adapter to both unmodified deployment workers on synthetic native data.
