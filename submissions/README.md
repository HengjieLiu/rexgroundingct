---
created: 2026-09-09
updated: 2026-09-09
status: active
---

# Submission register

Open [REGISTER.md](REGISTER.md) to identify a prediction set, its actual path,
model recipe, postprocessing and packaging status. [registry.json](registry.json)
is the authoritative editable record; REGISTER.md is generated from it.

Stable IDs identify prediction recipes, not upload attempts. For a1–d3, suffix 1 is
raw, 2 is eligible whole-lung support dilated 20 mm, and 3 is eligible
prompt-selected side/lobe support dilated 20 mm. Both postprocessed variants
come independently from the raw prediction. c1/c2/c3 alias the existing
baseline/whole_lung20/fine20 directories; no data is renamed or copied.

d11 and d12 are explicit additional method IDs in family d. d11 applies the
collaborator's frozen semantic-v1 to d1; d12 applies frozen strict semantic-v2
to d11. They do not follow the single-digit 1/2/3 convention. Their separate
SideExp005 val200 evidence does not establish test prediction or ZIP readiness.
The user authorized separate SideExp005 run
`r002_d1_test300_d11_d12_frozen_postprocessing` to generate both test variants
automatically after the val200 report is verified, regardless of scores.
This supersedes the earlier manual review gate. Their records transition to
the new producer after the val200 collector finishes; existing d1/d2/d3
completion records cannot cover them. ZIP creation is skipped by request,
recorded as `skipped_by_request` with null ZIP path and hash. Prediction
readiness still does not imply an upload.

## Commands

From the repository root, using the host Python standard library:

```bash
python submissions/manage.py check
python submissions/manage.py refresh
python submissions/manage.py render
python -m unittest discover -s submissions -p 'test_*.py'
```

`check` validates record consistency and that REGISTER.md matches the registry;
it does not need data mounts. `refresh` reads the mounted dataset, provenance,
completion and verification records, observes file counts, and checks available
ZIP hashes and contents. It updates observed artifact state and regenerates
REGISTER.md. Run it where the authoritative shared mount is available. It never
launches work, writes prediction files, packages ZIPs, or records an upload.
`render` regenerates only the human register after an intentional JSON edit.

Ready means the named completion/verification evidence passes and the expected
files exist. Refresh does not rerun NIfTI geometry validation or reread the
large model checkpoints; their recorded identities and validation provenance
remain linked. Hash mismatches or conflicting evidence are reported as
`conflict`; missing mounts/files cannot become ready. The last check timestamp
is an observation time, not a background monitor.

## Maintaining records

Add future recipes to `families`, definitions to `postprocessing` as needed,
and one record per stable submission ID. Do not reuse an ID for changed model
weights or postprocessing. Freeze checkpoint hashes and source records.
Prediction, ZIP and upload states are independent. Unknown values are null;
an empty `submission_history` means no upload has been recorded here.

After an actual upload, append an entry to that variant's `submission_history`:

```json
{
  "event_id": "a1-20260910-01",
  "submitted_name": "the exact name used on the submission site",
  "submitted_at_utc": "2026-09-10T12:00:00Z",
  "external_id": null,
  "zip_sha256": "the SHA-256 of the uploaded archive",
  "result_references": []
}
```

Use actual evidence, not the example values. Keep past upload events when
resubmitting; refresh preserves histories and manually recorded notes.
Unknown archive hashes may be null and must not be inferred from a later ZIP.

[leaderboard/](leaderboard/README.md) holds future result extraction and matching
conventions. [reviews/](reviews/README.md) holds interpretation and comparisons.
No leaderboard collection or upload is performed by this initial implementation.

Only small provenance records, public result snapshots and notes belong in Git.
NIfTIs, model weights, ZIPs, logs and large review images remain external. New
large review artifacts can use `/mnt/shengdata1/hengjie/submissions/rexgroundingct/`;
the existing prediction locations remain authoritative.
