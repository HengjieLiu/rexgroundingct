#!/usr/bin/env bash
set -euo pipefail

# Host-side, idempotent Exp021 -> Exp007 phase-3 gatekeeper.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
EXP021_DIR="${EXP021_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/021_voxtell_iso07_2d_nodule_specialist_from_exp017_e100}"
EXP007_DIR="${EXP007_DIR:-/mnt/shengdata1/hengjie/experiments/rexgroundingct/007_voxtell_cached_native_v123_e5_d4_ddp_bs4_update_matched}"
VAL200_JSON="${VAL200_JSON:-$REPO_ROOT/configs/evaluation/rexgroundingct_val200_seed20260723.json}"
IMAGE="${IMAGE:-rexgroundingct-voxtell:cu126}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-1800}"
POLL_ONCE="${POLL_ONCE:-0}"
NO_LAUNCH="${NO_LAUNCH:-0}"

AUTOSTART_DIR="${AUTOSTART_DIR:-$EXP007_DIR/reports/phase3_autostart}"
STATUS_JSON="$AUTOSTART_DIR/poller_status.json"
STATUS_MD="$AUTOSTART_DIR/poller_status.md"
LAUNCH_STATE_JSON="$AUTOSTART_DIR/phase3_launch_state.json"
LOG_FILE="$AUTOSTART_DIR/poller.log"
LOCK_FILE="$AUTOSTART_DIR/poller.lock"

mkdir -p "$AUTOSTART_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another Exp021 -> Exp007 phase-3 poller is already running" >&2
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_FILE"
}

atomic_json() {
  local path="$1" value="$2"
  python - "$path" "$value" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
tmp.write_text(sys.argv[2] + "\n")
os.replace(tmp, path)
PY
}

resolve_exp021_group() {
  if [[ -n "${EXP021_RUN_GROUP:-}" ]]; then
    printf '%s\n' "$EXP021_DIR/runs/$EXP021_RUN_GROUP"
    return 0
  fi
  if [[ -s "$EXP021_DIR/config/latest_run_group.txt" ]]; then
    local group
    group="$(head -n 1 "$EXP021_DIR/config/latest_run_group.txt")"
    printf '%s\n' "$EXP021_DIR/runs/$group"
    return 0
  fi
  if [[ -e "$EXP021_DIR/runs/latest" ]]; then
    readlink -f "$EXP021_DIR/runs/latest"
    return 0
  fi
  printf '%s\n' ""
}

inspect_exp021() {
  local group_dir="$1" output="$2"
  python - "$group_dir" "$EXP021_DIR" "$VAL200_JSON" "$output" <<'PY'
import json
import os
import sys
from pathlib import Path

group, exp_dir, val200, output = map(Path, sys.argv[1:])
run_dir = group / "ddp_bs4"
checks = []

def check(name, ok, detail):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

check("group_exists", group.is_dir(), group)
check("no_train_failed_marker", not (run_dir / ".train_failed").exists() and not (group / ".train_failed").exists(), run_dir)
check("train_complete_marker", (run_dir / ".train_complete").is_file(), run_dir / ".train_complete")
check("experiment_complete_marker", (group / ".experiment_complete").is_file(), group / ".experiment_complete")

checkpoint = run_dir / "checkpoints" / "checkpoint_update_010000.pth"
check("final_checkpoint_present", checkpoint.is_file() and checkpoint.stat().st_size > 0, checkpoint)

try:
    dataset = json.loads(val200.read_text())
    dataset_cases = len(dataset["test"])
    dataset_findings = sum(len(entry.get("findings", {})) for entry in dataset["test"])
except Exception as exc:
    check("val200_dataset_readable", False, exc)
    dataset_cases = dataset_findings = -1
else:
    check("val200_dataset_shape", dataset_cases == 200 and dataset_findings == 381, f"cases={dataset_cases} findings={dataset_findings}")

eval_dir = run_dir / "eval_epoch100_val200"
eval_json = eval_dir / "eval" / "val_quick_global_eval.json"
eval_summary = eval_dir / "reports" / "val_quick_global_eval_summary.json"
predictions = eval_dir / "predictions"
summary_data = {}
if eval_summary.is_file():
    try:
        summary_data = json.loads(eval_summary.read_text())
    except Exception as exc:
        check("final_eval_summary_readable", False, exc)
check("final_eval_json_present", eval_json.is_file(), eval_json)
check("final_eval_summary_present", eval_summary.is_file(), eval_summary)
check("final_eval_shape", summary_data.get("total_cases") == 200 and summary_data.get("total_findings") == 381, summary_data)
prediction_count = len(list(predictions.glob("*.nii.gz"))) if predictions.is_dir() else 0
check("final_prediction_count", prediction_count == 200, f"predictions={prediction_count}")

summary_path = exp_dir / "reports" / "nodule_specialist_summary.json"
specialist = {}
if summary_path.is_file():
    try:
        specialist = json.loads(summary_path.read_text())
    except Exception as exc:
        check("specialist_summary_readable", False, exc)
check("specialist_summary_complete", specialist.get("status") == "complete", summary_path)
milestone100 = specialist.get("milestones", {}).get("100", {})
check("specialist_e100_present", milestone100.get("evaluation") is not None, milestone100)

payload = {
    "group_dir": str(group),
    "run_dir": str(run_dir),
    "checked_artifacts": {
        "checkpoint": str(checkpoint),
        "eval_json": str(eval_json),
        "eval_summary": str(eval_summary),
        "predictions": str(predictions),
        "specialist_summary": str(summary_path),
    },
    "checks": checks,
    "complete": all(item["ok"] for item in checks),
}
path = Path(output)
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

read_launch_state() {
  if [[ -s "$LAUNCH_STATE_JSON" ]]; then
    cat "$LAUNCH_STATE_JSON"
  else
    printf '%s\n' '{}'
  fi
}

write_launch_state() {
  local status="$1" group_dir="$2" run_group="$3" container="$4" message="$5"
  python - "$LAUNCH_STATE_JSON" "$status" "$group_dir" "$run_group" "$container" "$message" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
old = json.loads(path.read_text()) if path.is_file() else {}
payload = dict(old)
payload["status"] = sys.argv[2]
for key, value in (("phase3_group_dir", sys.argv[3]), ("phase3_run_group", sys.argv[4]), ("container_name", sys.argv[5])):
    if value:
        payload[key] = value
payload["message"] = sys.argv[6]
payload["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
if sys.argv[2] == "launched" and "launched_at_utc" not in payload:
    payload["launched_at_utc"] = payload["updated_at_utc"]
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

write_status() {
  local status="$1" group_dir="$2" reason="$3" gate_json="$4" count="$5"
  python - "$STATUS_JSON" "$STATUS_MD" "$status" "$group_dir" "$reason" "$gate_json" "$LAUNCH_STATE_JSON" "$count" "$EXP021_DIR" "$EXP007_DIR" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

status_path, md_path = Path(sys.argv[1]), Path(sys.argv[2])
status, group, reason = sys.argv[3:6]
gate_path, launch_path = Path(sys.argv[6]), Path(sys.argv[7])
count = int(sys.argv[8])
exp021, exp007 = sys.argv[9:11]
gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {"complete": False, "checks": []}
launch = json.loads(launch_path.read_text()) if launch_path.is_file() else {}
payload = {
    "poller": "exp021_gated_exp007_phase3",
    "status": status,
    "reason": reason,
    "check_count": count,
    "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "poll_interval_seconds": int(os.environ.get("POLL_INTERVAL_SECONDS", "1800")),
    "exp021_dir": exp021,
    "exp021_group_dir": group,
    "exp021_gate": gate,
    "phase3": launch,
    "phase3_report_json": f"{exp007}/reports/phase3_status.json",
    "phase3_report_md": f"{exp007}/reports/phase3_status.md",
}

tmp = status_path.with_name(f".{status_path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, status_path)

lines = [
    "# Exp021 -> Exp007 Phase-3 Poller",
    "",
    f"Status: `{status}`",
    f"Last check: `{payload['checked_at_utc']}`",
    f"Check count: `{count}`; interval: `{payload['poll_interval_seconds']}` seconds",
    "",
    f"Reason: {reason}",
    "",
    "## Exp021 completion gate",
    "",
    f"- Observed group: `{group or 'not found'}`",
    f"- Gate result: `{'complete' if gate.get('complete') else 'incomplete'}`",
]
for item in gate.get("checks", []):
    lines.append(f"- {'PASS' if item['ok'] else 'WAIT'} `{item['name']}` — {item['detail']}")
lines.extend([
    "",
    "## Phase-3 launch",
    "",
    f"- State: `{launch.get('status', 'not_started')}`",
    f"- Run group: `{launch.get('phase3_group_dir', 'not started')}`",
    f"- Container: `{launch.get('container_name', 'not started')}`",
    f"- Message: {launch.get('message', 'n/a')}",
    "",
    "## Live phase-3 report",
    "",
    f"- Markdown: `{exp007}/reports/phase3_status.md`",
    f"- JSON: `{exp007}/reports/phase3_status.json`",
    f"- Poller log: `{Path(sys.argv[2]).with_name('poller.log')}`",
])
tmp_md = md_path.with_name(f".{md_path.name}.tmp.{os.getpid()}")
tmp_md.write_text("\n".join(lines) + "\n")
os.replace(tmp_md, md_path)
PY
}

check_count=0
if [[ -s "$STATUS_JSON" ]]; then
  check_count="$(python - "$STATUS_JSON" <<'PY'
import json
import sys
from pathlib import Path
try:
    print(int(json.loads(Path(sys.argv[1]).read_text()).get("check_count", 0)))
except Exception:
    print(0)
PY
)"
fi

while true; do
  check_count=$((check_count + 1))
  group_dir="$(resolve_exp021_group)"
  gate_json="$AUTOSTART_DIR/exp021_gate.json"
  if [[ -n "$group_dir" ]]; then
    inspect_exp021 "$group_dir" "$gate_json"
  else
    atomic_json "$gate_json" '{"complete": false, "group_dir": "", "checks": [{"name": "latest_run_group", "ok": false, "detail": "No Exp021 latest run group is available"}]}'
  fi

  launch_status="$(python - "$LAUNCH_STATE_JSON" <<'PY'
import json
import sys
from pathlib import Path
try:
    print(json.loads(Path(sys.argv[1]).read_text()).get("status", "not_started"))
except Exception:
    print("not_started")
PY
)"
  launch_group_dir="$(python - "$LAUNCH_STATE_JSON" <<'PY'
import json
import sys
from pathlib import Path
try:
    print(json.loads(Path(sys.argv[1]).read_text()).get("phase3_group_dir", ""))
except Exception:
    print("")
PY
)"

  status="waiting_exp021"
  reason="Exp021 has not passed the validated completion gate"

  if [[ "$launch_status" == "complete" ]]; then
    status="complete"
    reason="Exp007 phase 3 was already completed"
  elif [[ "$launch_status" == "launched" || "$launch_status" == "running" || "$launch_status" == "launching" ]]; then
    if [[ -f "$launch_group_dir/.experiment_complete" ]]; then
      status="complete"
      reason="Exp007 phase 3 completed"
      write_launch_state "complete" "$launch_group_dir" "$(basename "$launch_group_dir")" "" "Phase 3 completed"
    elif [[ -f "$launch_group_dir/.train_failed" ]]; then
      status="failed"
      reason="Exp007 phase 3 reported .train_failed"
    else
      if [[ "$launch_status" == "launching" ]]; then
        status="phase3_launching"
        reason="A phase-3 launch intent already exists; no duplicate launch will be attempted"
      else
        status="phase3_running"
        reason="Exp007 phase 3 has been launched and is still in progress"
      fi
    fi
  elif [[ "$launch_status" == "failed" ]]; then
    status="failed"
    reason="Phase-3 launch previously failed; inspect the poller log"
  elif [[ "$NO_LAUNCH" == "1" ]] && python - "$gate_json" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("complete") else 1)
PY
  then
    status="ready_to_launch"
    reason="Exp021 passed; NO_LAUNCH=1 is preventing launch"
  elif python - "$gate_json" <<'PY'
import json
import sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("complete") else 1)
PY
  then
    if ! docker info >/dev/null 2>&1; then
      status="waiting_for_docker_gpu"
      reason="Docker daemon is unavailable to the poller account"
      log "$reason"
    elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      status="waiting_for_docker_gpu"
      reason="Required image is unavailable: $IMAGE"
      log "$reason"
    else
      phase3_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
      phase3_run_group="exp007_phase3_from_abs_e200_$phase3_timestamp"
      phase3_group_dir="$EXP007_DIR/runs/$phase3_run_group"
      container_name="rex007_phase3_from_abs_e200_$phase3_timestamp"
      write_launch_state "launching" "$phase3_group_dir" "$phase3_run_group" "$container_name" "Launching after Exp021 completion"
      log "Exp021 passed; launching Exp007 phase 3 group=$phase3_run_group container=$container_name"
      if REPO_ROOT="$REPO_ROOT" EXP_DIR="$EXP007_DIR" RUN_GROUP="$phase3_run_group" \
        CONTAINER_NAME="$container_name" IMAGE="$IMAGE" DETACH=1 \
        bash "$SCRIPT_DIR/run_007_phase3_from_abs_e200_docker.sh" >>"$LOG_FILE" 2>&1; then
        write_launch_state "launched" "$phase3_group_dir" "$phase3_run_group" "$container_name" "Detached phase-3 container launched"
        status="phase3_running"
        reason="Exp021 passed and phase 3 was launched"
      else
        write_launch_state "failed" "$phase3_group_dir" "$phase3_run_group" "$container_name" "Docker phase-3 launch command failed"
        status="failed"
        reason="Docker phase-3 launch command failed; inspect the poller log"
      fi
    fi
  fi

  write_status "$status" "$group_dir" "$reason" "$gate_json" "$check_count"
  log "status=$status exp021_group=${group_dir:-missing}"

  if [[ "$POLL_ONCE" == "1" || "$status" == "complete" || "$status" == "failed" ]]; then
    exit 0
  fi
  sleep "$POLL_INTERVAL_SECONDS"
done
