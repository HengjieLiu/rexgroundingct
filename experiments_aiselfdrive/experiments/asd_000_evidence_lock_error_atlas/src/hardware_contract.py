"""Pure revision-2 hardware inventory contracts shared by host and Docker."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


def validate_cuda_inventory(
    observed_names: Sequence[str],
    *,
    expected_count: int,
    expected_name: str,
    require_exact: bool,
) -> dict[str, Any]:
    names = list(observed_names)
    observed_count = len(names)
    if expected_count <= 0 or not expected_name:
        raise ValueError("expected CUDA inventory is malformed")
    if observed_count <= 0 or any(not name for name in names):
        raise RuntimeError("Docker exposes no valid CUDA devices")
    exact = observed_count == expected_count and all(
        name == expected_name for name in names
    )
    if require_exact and not exact:
        raise RuntimeError(
            "Docker CUDA inventory mismatch: "
            f"count={observed_count}, names={names!r}"
        )
    return {
        "visible_cuda_device_count": observed_count,
        "visible_cuda_device_names": names,
        "expected_cuda_device_count": expected_count,
        "expected_cuda_device_name": expected_name,
        "cuda_inventory_exact": exact,
    }


def atomic_runtime_durability_probe(root: str | Path) -> dict[str, Any]:
    """Prove atomic write, fsync, readback, and deletion at a runtime root."""

    runtime_root = Path(root)
    if not runtime_root.is_absolute():
        raise ValueError("runtime durability probe root must be absolute")
    runtime_root.mkdir(parents=True, exist_ok=True)
    resolved_root = runtime_root.resolve(strict=True)
    payload = b"rexgroundingct-asd000-runtime-durability-probe-v1\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=".asd000-runtime-probe-",
        suffix=".tmp",
        dir=resolved_root,
        delete=False,
    )
    temporary = Path(handle.name)
    committed = temporary.with_suffix(".committed")
    directory_descriptor: int | None = None
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, committed)
        directory_descriptor = os.open(
            resolved_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(directory_descriptor)
        with committed.open("rb", buffering=0) as stream:
            observed = stream.read(len(payload) + 1)
        if observed != payload:
            raise RuntimeError("external runtime durability probe readback mismatch")
        committed.unlink()
        os.fsync(directory_descriptor)
        if temporary.exists() or committed.exists():
            raise RuntimeError("external runtime durability probe cleanup failed")
    except OSError as exc:
        raise RuntimeError(
            f"external runtime durability probe failed at {resolved_root}: {exc}"
        ) from exc
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        for candidate in (temporary, committed):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
    return {
        "status": "passed",
        "runtime_root": str(resolved_root),
        "probe_contract": "atomic_replace_file_fsync_directory_fsync_read_delete_v1",
        "probe_bytes": len(payload),
        "file_fsync": True,
        "atomic_replace": True,
        "directory_fsync_after_replace": True,
        "readback_exact": True,
        "delete_verified": True,
        "directory_fsync_after_delete": True,
    }


__all__ = ["atomic_runtime_durability_probe", "validate_cuda_inventory"]
