#!/usr/bin/env bash
# Build the example app for an iPhone SE simulator, run the deterministic
# --memory-test suite, and assert that phys_footprint stays within budget.
#
# Usage:
#   scripts/memcheck.sh                       # default thresholds + sim
#   PEAK_MB=4500 scripts/memcheck.sh          # override peak budget
#   LEAK_MB=120 scripts/memcheck.sh           # post-run residual budget
#   DEVICE_NAME="iPhone 16" scripts/memcheck.sh
#   XCTRACE=1 scripts/memcheck.sh             # also record an Allocations trace
#
# Important: simulator memory is NOT a faithful proxy for device memory.
# CoreML's GPU/MPS working set counts against phys_footprint differently
# than ANE allocations on real hardware. Use this script to:
#   1. Detect leaks: teardown footprint should return near baseline.
#   2. Detect regressions: peak shouldn't grow run-over-run.
#   3. Compare optimizations: same suite before/after a change.
# For absolute "won't get jetsamed on iPhone SE", run on real hardware.
#
# Exit codes:
#   0  pass
#   1  budget exceeded
#   2  simulator setup failed
#   3  build failed
#   4  built app not found
#   5  test run timed out
#   10 result markers not found in log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${REPO_ROOT}/Examples/KokoroApp"
PROJECT="${APP_DIR}/KokoroApp.xcodeproj"
SCHEME="KokoroApp"
BUNDLE_ID="com.kokoro.app"

DEVICE_NAME="${DEVICE_NAME:-iPhone SE (3rd generation)}"
# Simulator-realistic defaults. CoreML on simulator uses CPU/Metal, which
# has a much larger working set than ANE on real device. These are picked
# to flag regressions; real-device numbers are typically 10-20x lower.
PEAK_MB="${PEAK_MB:-4500}"
DELTA_MB="${DELTA_MB:-4500}"
LEAK_MB="${LEAK_MB:-150}"
TIMEOUT_S="${TIMEOUT_S:-180}"
XCTRACE="${XCTRACE:-0}"
ARTIFACTS_DIR="${REPO_ROOT}/.memcheck"
mkdir -p "$ARTIFACTS_DIR"

log() { echo "[memcheck] $*"; }

resolve_device() {
  local name="$1"
  xcrun simctl list devices available --json \
    | python3 -c "
import json,sys
data=json.load(sys.stdin)
target=sys.argv[1]
for runtime, devs in data['devices'].items():
    for d in devs:
        if d['name']==target and d['isAvailable']:
            print(d['udid']); sys.exit(0)
sys.exit(1)
" "$name" 2>/dev/null
}

UDID="$(resolve_device "$DEVICE_NAME" || true)"
if [ -z "$UDID" ]; then
  log "Simulator \"$DEVICE_NAME\" not found, attempting to create it..."
  DEVICE_TYPE_ID="$(xcrun simctl list devicetypes | grep -F "$DEVICE_NAME" | grep -oE 'com\.apple\.CoreSimulator\.SimDeviceType\.[A-Za-z0-9-]+' | head -1)"
  RUNTIME_ID="$(xcrun simctl list runtimes | grep -E '^iOS' | grep -oE 'com\.apple\.CoreSimulator\.SimRuntime\.iOS-[0-9-]+' | tail -1)"
  if [ -z "$DEVICE_TYPE_ID" ] || [ -z "$RUNTIME_ID" ]; then
    log "ERROR: cannot create — device type \"$DEVICE_NAME\" or iOS runtime not found"
    log "Available device types matching SE/iPhone:"
    xcrun simctl list devicetypes | grep -iE "iPhone SE|iPhone 1[6-9]|iPhone Air" | sed 's/^/  /'
    exit 2
  fi
  UDID="$(xcrun simctl create "$DEVICE_NAME" "$DEVICE_TYPE_ID" "$RUNTIME_ID")"
  log "Created $DEVICE_NAME ($UDID) on $RUNTIME_ID"
fi
log "Device: $DEVICE_NAME ($UDID)"

state="$(xcrun simctl list devices --json | python3 -c "
import json,sys
data=json.load(sys.stdin)
udid=sys.argv[1]
for runtime, devs in data['devices'].items():
    for d in devs:
        if d['udid']==udid:
            print(d['state']); sys.exit(0)
" "$UDID")"
if [ "$state" != "Booted" ]; then
  log "Booting simulator..."
  xcrun simctl boot "$UDID"
  sleep 3
fi

if [ ! -d "$PROJECT" ]; then
  log "Xcode project missing, generating from project.yml..."
  if ! command -v xcodegen >/dev/null 2>&1; then
    log "ERROR: xcodegen not found; install with 'brew install xcodegen' or commit a tracked project"
    exit 3
  fi
  (cd "$APP_DIR" && xcodegen generate > "${ARTIFACTS_DIR}/xcodegen.log" 2>&1) \
    || { log "ERROR: xcodegen failed — see ${ARTIFACTS_DIR}/xcodegen.log"; tail -20 "${ARTIFACTS_DIR}/xcodegen.log"; exit 3; }
fi

log "Building app..."
DERIVED="${ARTIFACTS_DIR}/DerivedData"
xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration Debug \
  -destination "platform=iOS Simulator,id=$UDID" \
  -derivedDataPath "$DERIVED" \
  -quiet \
  build > "${ARTIFACTS_DIR}/build.log" 2>&1 \
  || { log "ERROR: build failed — see ${ARTIFACTS_DIR}/build.log"; tail -30 "${ARTIFACTS_DIR}/build.log"; exit 3; }

APP_PATH="$(find "$DERIVED/Build/Products" -name "${SCHEME}.app" -type d -maxdepth 4 | head -1)"
if [ -z "$APP_PATH" ]; then
  log "ERROR: built app not found under $DERIVED/Build/Products"
  exit 4
fi
log "App: $APP_PATH"

log "Installing..."
xcrun simctl install "$UDID" "$APP_PATH"

LOG_FILE="${ARTIFACTS_DIR}/run.log"
rm -f "$LOG_FILE"

log "Running --memory-test (timeout ${TIMEOUT_S}s)..."
(
  xcrun simctl launch --console-pty --terminate-running-process \
    "$UDID" "$BUNDLE_ID" --memory-test \
    > "$LOG_FILE" 2>&1
) &
LAUNCH_PID=$!

SECONDS_WAITED=0
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
  if [ "$SECONDS_WAITED" -ge "$TIMEOUT_S" ]; then
    log "ERROR: timeout, terminating app"
    xcrun simctl terminate "$UDID" "$BUNDLE_ID" 2>/dev/null || true
    kill "$LAUNCH_PID" 2>/dev/null || true
    exit 5
  fi
  sleep 1
  SECONDS_WAITED=$((SECONDS_WAITED + 1))
done

if [ "$XCTRACE" = "1" ]; then
  TRACE_OUT="${ARTIFACTS_DIR}/allocations.trace"
  rm -rf "$TRACE_OUT"
  log "Recording xctrace Allocations (this is slow)..."
  xcrun xctrace record \
    --template "Allocations" \
    --device "$UDID" \
    --output "$TRACE_OUT" \
    --launch -- "$APP_PATH/${SCHEME}" --memory-test \
    --time-limit "${TIMEOUT_S}s" \
    > "${ARTIFACTS_DIR}/xctrace.log" 2>&1 \
    || log "WARN: xctrace returned non-zero — see ${ARTIFACTS_DIR}/xctrace.log"
  log "Trace: $TRACE_OUT (open in Instruments.app)"
fi

JSON_OUT="${ARTIFACTS_DIR}/result.json"
python3 - "$LOG_FILE" "$JSON_OUT" <<'PY'
import sys, re
log_path, out_path = sys.argv[1], sys.argv[2]
with open(log_path, "r", errors="replace") as f:
    text = f.read()
m = re.search(r"MEMORY_TEST_RESULT_START\s*\n(.*?)\nMEMORY_TEST_RESULT_END", text, re.S)
if not m:
    sys.stderr.write("no result markers found in log\n")
    sys.stderr.write(text[-2000:])
    sys.exit(10)
with open(out_path, "w") as out:
    out.write(m.group(1))
print(out_path)
PY

if ! python3 - "$JSON_OUT" "$PEAK_MB" "$DELTA_MB" "$LEAK_MB" <<'PY'
import json, sys
path = sys.argv[1]
peak_budget = float(sys.argv[2])
delta_budget = float(sys.argv[3])
leak_budget = float(sys.argv[4])
data = json.load(open(path))
peak = data["peakMB"]
delta = data["peakDeltaMB"]
baseline = data["baselineMB"]
duration = data["durationSeconds"]
events = data["events"]

teardown = next((e for e in events if e["name"] == "teardown"), None)
post_load = next((e for e in events if e["name"] == "post_engine_load"), None)
residual = (teardown["mb"] - baseline) if teardown else None

error_events = [e for e in events if e["name"] in ("error", "chunk_failed")]
case_starts = sum(1 for e in events if e["name"] == "case_start")
case_ends = sum(1 for e in events if e["name"] == "case_end")
first_buffers = sum(1 for e in events if e["name"] == "first_buffer")

print(f"baseline:       {baseline:7.1f} MB")
if post_load:
    print(f"engine load:   +{post_load['mb'] - baseline:7.1f} MB  (resident after model load)")
print(f"peak:           {peak:7.1f} MB  (budget {peak_budget:.0f} MB)")
print(f"peak delta:    +{delta:7.1f} MB  (budget {delta_budget:.0f} MB)")
if teardown:
    print(f"residual:      +{residual:7.1f} MB  (budget {leak_budget:.0f} MB) — should be near 0")
print(f"duration:       {duration:7.1f} s")
print(f"events:         {len(events)}")
print(f"samples:        {len(data['samples'])}")
print()
print("Lifecycle events (mb):")
for ev in events:
    detail = f"  [{ev['detail']}]" if ev.get("detail") else ""
    print(f"  {ev['timeSeconds']:6.2f}s  {ev['name']:24s} {ev['mb']:7.1f}{detail}")
print()

failures = []
if error_events:
    names = ", ".join(f"{e['name']}({(e.get('detail') or '')[:60]})" for e in error_events[:5])
    failures.append(f"runtime errors logged: {names}")
if case_starts == 0:
    failures.append("no synthesis cases ran")
elif case_ends != case_starts:
    failures.append(f"only {case_ends}/{case_starts} cases completed")
elif first_buffers != case_starts:
    failures.append(f"only {first_buffers}/{case_starts} cases produced audio")
if peak > peak_budget:
    failures.append(f"peak {peak:.1f} MB > budget {peak_budget:.0f} MB")
if delta > delta_budget:
    failures.append(f"peak delta {delta:.1f} MB > budget {delta_budget:.0f} MB")
if teardown and residual > leak_budget:
    failures.append(f"residual {residual:.1f} MB > leak budget {leak_budget:.0f} MB (possible leak)")
if failures:
    print("FAIL: " + "; ".join(failures))
    sys.exit(1)
else:
    print(f"Within budget ({case_ends}/{case_starts} cases, {first_buffers} produced audio).")
    sys.exit(0)
PY
then
  log "FAIL: memory budget exceeded"
  exit 1
fi

log "PASS"
