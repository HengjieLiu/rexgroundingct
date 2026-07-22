---
created: 2026-07-22
updated: 2026-07-22
status: active
---

# ReXGroundingCT Prompt Illustration

This folder documents the English free-text prompts used in the ReXGroundingCT
train/val/test splits and provides Chinese translations for prompt
understanding.

Source metadata:

```text
/data/hengjie/datasets/rexgroundingct/MICCAI_challenge_dataset.json
```

These translations are for research reading, anatomy learning, and prompt
analysis. They are not certified clinical report translations.

## Split Counts

| Split | Cases | Prompt instances | Unique prompts |
| --- | ---: | ---: | ---: |
| train | 2992 | 7687 | 6148 |
| val | 200 | 381 | 365 |
| test | 300 | 582 | 542 |
| total | 3492 | 8650 | 6926 |

## Folder Map

- `unique_prompts_bilingual.md`
  - One row per unique English prompt.
  - Includes Chinese translation and train/val/test/total appearance counts.

- `prompt_instances_bilingual.md`
  - One row per prompt occurrence.
  - Preserves split, case name, finding ID, category, and entity count when
    available.

- `vocabulary_bilingual.md`
  - Structured glossary of disease, anatomy, location, morphology, size,
    severity, temporal, and uncertainty terms.
  - Includes raw surface variants, split counts, and example prompts.

## Translation Style

Chinese translations use radiology-oriented phrasing and keep important English
terms in parentheses. The translation pass is deterministic and glossary-based,
so uncommon long prompts may remain partially bilingual when an English fragment
does not map cleanly to the glossary.
