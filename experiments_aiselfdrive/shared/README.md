# Shared self-drive assets

This directory contains immutable manifests, the expected hardware profile,
small reusable source modules, and shared tests for the self-drive portfolio.
Experiment-specific mutable outputs do not belong here.

`hardware_profile.yaml` pins both human-readable Docker tags and the observed
local image IDs. Anatomy stages also verify a deterministic sorted-file hash
of the mounted TotalSegmentator weights; a matching tag alone is not accepted.
The control-plane lock manifest is regenerated only after an audited registry
or plan revision and excludes mutable experiment state.
