# Frozen Source Snapshots

This directory contains hash-named, byte-for-byte source snapshots needed to
validate the immutable Side Experiment 002 candidate roster when a live
canonical experiment file later changes.

The exporter always checks the canonical path first. It uses a snapshot only
when the canonical file no longer has the SHA256 frozen in
`candidate_manifest.json`, and it independently verifies that the snapshot has
the expected hash.

The Exp007 snapshot preserves the configuration used by the epoch-50, epoch-75,
and epoch-100 ensemble candidates. The canonical Exp007 configuration now also
contains continuation-training metadata and must not be overwritten.
