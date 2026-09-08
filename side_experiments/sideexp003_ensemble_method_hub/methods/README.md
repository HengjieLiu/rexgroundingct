---
created: 2026-09-07
updated: 2026-09-07
status: active
---

# Ensemble Methods

Each algorithm owns one folder and one `method_spec.md`. Runs are immutable
children named `rNNN_<roster-slug>` and mirror the same relative path under the
SideExp003 `/mnt` runtime root.

Do not add new logic to an old method/run. Add a new method folder and spec,
then create a new run from the templates. The two initial OOF methods are
specified but have not been executed. `uniform_global` and
`caruana_replacement_global` also support explicitly labelled metrics-only
full-val diagnostics; these never become formal recipes.

`caruana_replacement_seeded_scoped` is the separately authorized top-16
same-val diagnostic. It freezes different category-ranked K=4 seeds for
`all/2a/2b/2c/2d`, performs replacement additions through K=16, and shares
source reads across all five scopes.
