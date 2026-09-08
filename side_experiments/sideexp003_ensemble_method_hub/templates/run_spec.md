---
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: draft
---

# RUN_ID — METHOD_NAME

## Decision gates

- [ ] roster confirmed by user
- [ ] cache audit passed
- [ ] `Mmax` confirmed by user
- [ ] search complete
- [ ] final `M` confirmed by user
- [ ] final recipe materialized and audited

## Frozen identity

Mirror every value in `run_spec.json`: roster/catalog/dataset/fold/cache hashes,
`N`, `Mmax`, method parameters, tie-break, commands, code revision, and tracked
plus runtime output locations.

## Results and decision

Link results without changing frozen inputs. Record diagnostic and OOF metrics
separately. The final decision must quote the user's confirmed `M`.
