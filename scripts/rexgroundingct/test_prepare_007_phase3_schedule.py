from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_007_phase3_schedule import materialize_slice  # noqa: E402


def write_schedule(path: Path, count: int, prefix: str = "case") -> None:
    path.write_text(
        "".join(
            json.dumps({
                "event_index": index,
                "case_name": f"{prefix}_{index}",
                "stats": {"events": 1},
            }, sort_keys=True) + "\n"
            for index in range(count)
        )
    )


def test_phase3_slice_preserves_source_indices_and_reindexes(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    prior = tmp_path / "prior.jsonl"
    extended = tmp_path / "extended.jsonl"
    output = tmp_path / "phase3.jsonl"
    manifest = tmp_path / "phase3.manifest.json"
    write_schedule(original, 4)
    write_schedule(prior, 8)
    write_schedule(extended, 12)

    payload = materialize_slice(original, prior, extended, output, manifest, 8, 4, 4, 8)

    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert [event["event_index"] for event in events] == [0, 1, 2, 3]
    assert [event["source_event_index"] for event in events] == [8, 9, 10, 11]
    assert payload["source_event_start_inclusive"] == 8
    assert payload["source_event_stop_exclusive"] == 12
    assert json.loads(manifest.read_text())["events"] == 4


def test_phase3_slice_rejects_a_changed_prior_prefix(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    prior = tmp_path / "prior.jsonl"
    extended = tmp_path / "extended.jsonl"
    write_schedule(original, 4)
    write_schedule(prior, 8)
    write_schedule(extended, 12)
    changed = json.loads(extended.read_text().splitlines()[2])
    changed["case_name"] = "changed"
    lines = extended.read_text().splitlines()
    lines[2] = json.dumps(changed, sort_keys=True)
    extended.write_text("\n".join(lines) + "\n")

    try:
        materialize_slice(original, prior, extended, tmp_path / "out.jsonl", tmp_path / "manifest.json", 8, 4, 4, 8)
    except ValueError as exc:
        assert "prefix mismatch" in str(exc)
    else:
        raise AssertionError("Expected changed schedule prefix to be rejected")
