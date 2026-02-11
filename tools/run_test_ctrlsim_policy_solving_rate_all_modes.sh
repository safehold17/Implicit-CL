#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

SCENARIO_INDEX_PATH="/home/chen/workspace/dcd-ctrlsim/data/scenarios_index_filtered.json"
SCENARIO_DATA_DIR="/home/chen/data/nocturne_waymo/formatted_json_v2_no_tl_train"
PREPROCESS_DIR="/home/chen/data/preprocess/train"
VEHICLE_MAP_PATH="/home/chen/workspace/dcd-ctrlsim/data/vehicle_map_filtered.json"
CHECKPOINT_PATH="checkpoints/model.ckpt"
NUM_PROCESSES=4
NUM_EPISODES=200
OUTPUT_DIR="/home/chen/logs"
BASE_XPID="test-ctrlsim-metrics"
TILT_RANGE_MIN=-10
TILT_RANGE_MAX=-25

MODES=(
  "per_vehicle"
  "global"
  "ego"
  "none"
)

FAILED=()

echo "Repo root: ${REPO_ROOT}"
echo "Python bin: ${PYTHON_BIN}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Base xpid: ${BASE_XPID}"
echo

for mode in "${MODES[@]}"; do
  xpid="${BASE_XPID}-${mode}"
  echo "============================================================"
  echo "Running mode: ${mode}"
  echo "XPID: ${xpid}"
  echo "============================================================"

  if "${PYTHON_BIN}" tools/test_ctrlsim_policy_solving_rate.py \
    --scenario_index_path "${SCENARIO_INDEX_PATH}" \
    --scenario_data_dir "${SCENARIO_DATA_DIR}" \
    --preprocess_dir "${PREPROCESS_DIR}" \
    --vehicle_map_path "${VEHICLE_MAP_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --tilting_mode "${mode}" \
    --tilt_range "${TILT_RANGE_MIN}" "${TILT_RANGE_MAX}" \
    --num_processes "${NUM_PROCESSES}" \
    --num_episodes "${NUM_EPISODES}" \
    --show_vehicle_ids \
    --output_dir "${OUTPUT_DIR}" \
    --xpid "${xpid}" \
    --verbose; then
    echo "[OK] mode=${mode}"
  else
    code=$?
    echo "[FAIL] mode=${mode}, exit_code=${code}"
    FAILED+=("${mode}:${code}")
  fi

  echo
done

echo "===================== SUMMARY ====================="
for mode in "${MODES[@]}"; do
  xpid="${BASE_XPID}-${mode}"
  metrics_path="${OUTPUT_DIR}/${xpid}/metrics.csv"
  echo "mode=${mode} metrics=${metrics_path}"
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo
  echo "Failed runs:"
  for item in "${FAILED[@]}"; do
    echo "  - ${item}"
  done
  exit 1
fi

echo
echo "All 4 runs completed successfully."
