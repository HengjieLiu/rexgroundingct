# Challenge Information Archive

This folder contains source-faithful Markdown transcriptions and immutable evidence snapshots for ReXGroundingCT. Raw responses are authoritative whenever an interactive website cannot be represented fully in Markdown.

## Latest Extraction

- UTC: `2026-07-22T00:13:43Z`
- Local: `2026-07-21T17:13:43-07:00`
- Snapshot: [`snapshots/2026-07-22T001343Z/`](snapshots/2026-07-22T001343Z/)
- Manifest: [`snapshots/2026-07-22T001343Z/manifest.json`](snapshots/2026-07-22T001343Z/manifest.json)
- ReXrank `gh-pages` commit observed during extraction: `a0f39149ef7eee3b23bb2f674fe91fbeb18d40ba`

## Current Documents

| Record | Official source | Literal Markdown | Raw evidence | SHA-256 |
| --- | --- | --- | --- | --- |
| ReXGroundingCT | [https://rexrank.ai/ReXGroundingCT/](https://rexrank.ai/ReXGroundingCT/) | [rexgroundingct.md](rexgroundingct.md) | [`raw`](snapshots/2026-07-22T001343Z/raw/rexrank.ai/ReXGroundingCT/index.html) | `b5818cbbea4e...` |
| ReXGroundingCT Challenge @ MICCAI 2026 | [https://rexrank.ai/ReXGroundingCT/challenge.html](https://rexrank.ai/ReXGroundingCT/challenge.html) | [rexgroundingct_challenge.md](rexgroundingct_challenge.md) | [`raw`](snapshots/2026-07-22T001343Z/raw/rexrank.ai/ReXGroundingCT/challenge.html) | `3b1264e89db2...` |
| ReXrankCT Submission Guideline | [https://rexrank.ai/explore/submission_guideline_ct.html](https://rexrank.ai/explore/submission_guideline_ct.html) | [rexrankct_submission_guideline.md](rexrankct_submission_guideline.md) | [`raw`](snapshots/2026-07-22T001343Z/raw/rexrank.ai/explore/submission_guideline_ct.html) | `771a84f15ea9...` |
| ReXGroundingCT Dataset Card | [https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT) | [huggingface_dataset_card.md](huggingface_dataset_card.md) | [`raw`](snapshots/2026-07-22T001343Z/raw/huggingface.co/datasets/rajpurkarlab/ReXGroundingCT.html) | `e4a0ecee7330...` |
| ReXGroundingCT Data Generation Repository | [https://github.com/rajpurkarlab/ReXGroundingCT](https://github.com/rajpurkarlab/ReXGroundingCT) | [github_dataset_generation.md](github_dataset_generation.md) | [`raw`](snapshots/2026-07-22T001343Z/raw/raw.githubusercontent.com/rajpurkarlab/ReXGroundingCT/main/README.md) | `587e0588b050...` |
| ReXGroundingCT Paper Record | [https://arxiv.org/abs/2507.22030](https://arxiv.org/abs/2507.22030) | [arxiv_paper_record.md](arxiv_paper_record.md) | [`raw`](snapshots/2026-07-22T001343Z/raw/arxiv.org/abs/2507.22030.html) | `2feb8dc0e8c0...` |
| MICCAI 2026 Challenge Listing | [https://conferences.miccai.org/2026/en/challenges.asp](https://conferences.miccai.org/2026/en/challenges.asp) | [miccai_2026_challenge_listing.md](miccai_2026_challenge_listing.md) | [`raw`](snapshots/2026-07-22T001343Z/raw/conferences.miccai.org/2026/en/challenges.asp.html) | `81337ec4a9ad...` |

[`SUMMARY.md`](SUMMARY.md) is a concise, source-linked digest. It is deliberately separate from the literal transcriptions above.

## Supporting Data

| Artifact | Snapshot | Bytes | SHA-256 |
| --- | --- | --- | --- |
| rexgroundingct_model_results | [csv](snapshots/2026-07-22T001343Z/raw/rexrank.ai/ReXGroundingCT/ReXGroundingCT.csv) | 743 | `b8cd4c063002c598876ea8e227e396258c651da73c7dc69e0ae46d012daddc8e` |
| rexgroundingct_per_category_results | [csv](snapshots/2026-07-22T001343Z/raw/rexrank.ai/ReXGroundingCT/per_category_results.csv) | 4336 | `4ca83e4165e931d8d54b6abea5e67aea89e16073e633c8a25b94911ef2bec073` |
| rexgroundingct_category_counts | [csv](snapshots/2026-07-22T001343Z/raw/rexrank.ai/ReXGroundingCT/category_counts.csv) | 86 | `b9ba170f7a26ea91799088e3f434807141cdde1b5338dde9ad668171e9f455b5` |
| rexrank_gh_pages_branch | [json](snapshots/2026-07-22T001343Z/raw/api.github.com/repos/rajpurkarlab/ReXrank/branches/gh-pages.json) | 3794 | `b040c1bfd943b0c47d1102d8f966eada9cd6031c2bfea93da8d1da971dbeeaad` |
| challenge_public_leaderboard | [json](snapshots/2026-07-22T001343Z/raw/firestore.googleapis.com/rexgrounding-challenge/leaderboard.json) | 125456 | `d33ccef8f34dfb2e242eb48f648f88fad8d40948db972f37c4748bcb2de47f7a` |

The static ReXGroundingCT page embeds leaderboard values and is also supported by the archived CSV files. The MICCAI challenge leaderboard is runtime-loaded, so its public Firestore response is stored separately as JSON.

## Known Source Differences

- The challenge page states that its authenticated Register & Submit form is the only MICCAI submission channel. The linked guideline still contains instructions to submit by email. Both texts are preserved unchanged; the challenge page explicitly describes the guideline as a prediction-format reference rather than a separate submission route.
- The main ReXrank page describes a base 100-scan test set. The MICCAI challenge describes 200 validation and 300 test scans. The Hugging Face dataset card records the MICCAI split as an addition while the base dataset remains unchanged.
- The established model benchmark and the MICCAI public-split leaderboard are different leaderboards and remain separate in this archive.

## Gated Resources

The challenge links to resources inside the gated Hugging Face dataset. This updater records the official links but does not authenticate, bypass access controls, or copy unavailable content:

- [`rexrank_eval.py`](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/blob/main/rexrank_eval.py)
- [`MICCAI_challenge_dataset.json`](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/blob/main/MICCAI_challenge_dataset.json)

## Updating

Run from the repository root:

```powershell
python challenge_info/update_challenge_info.py
python challenge_info/update_challenge_info.py --verify-latest
```

Each successful update creates a new UTC-timestamped snapshot and refreshes the top-level Markdown files. Downloads, parsing, and validation finish before current files are promoted. A required-source failure exits nonzero and leaves the current documents unchanged.

Refresh before making submission decisions and regularly while the challenge leaderboard is active. Add future static official sources to [`sources.json`](sources.json). Dynamic sources require explicit code so a failed runtime request cannot be mistaken for an empty result.

## Integrity Policy

- Raw response bytes are stored before conversion and hashed with SHA-256.
- HTTP status, final URL, content type, ETag, Last-Modified, retrieval time, size, and hash are recorded in the manifest when available.
- Source wording, spelling, links, and displayed numeric precision are retained. Archive-generated notes are labeled explicitly.
- Only the intentionally public Firestore `leaderboard` collection is read. Private teams, submissions, authentication data, and withheld scores are outside scope.
- Full dataset binaries and paper PDFs are outside scope.

## Update Warnings

- None.
