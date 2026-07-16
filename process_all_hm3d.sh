#!/bin/bash
set -uo pipefail

# Process every locally downloaded HM3D Zed trajectory. Git-LFS pointer files
# are skipped so this script never triggers an accidental multi-terabyte fetch.

SCRIPT_DIR=/home/rlan/projects/vln-data-prep
TRAJ_DIR=/ssd5/datasets/InternData-N1/vln_n1/traj_data/hm3d_zed
LOG_DIR=/tmp/opencode/hm3d_batch_logs
OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/hm3d
mkdir -p "$LOG_DIR"

SUMMARY="${LOG_DIR}/batch_summary.log"
echo "Batch start: $(date)" | tee "$SUMMARY"

while IFS= read -r archive; do
    scene=$(basename "$archive" .tar.gz)
    if head -n 1 "$archive" | grep -q 'git-lfs.github.com/spec'; then
        echo "[$(date +%H:%M:%S)] SKIP  $scene (Git-LFS pointer)" | tee -a "$SUMMARY"
        continue
    fi

    echo "[$(date +%H:%M:%S)] START $scene" | tee -a "$SUMMARY"
    if bash "${SCRIPT_DIR}/run_pipeline_hm3d.sh" "$scene" > "${LOG_DIR}/${scene}.log" 2>&1; then
        n_rgb=$(find "${OUTPUT_ROOT}/${scene}/videos/chunk-000/observation.images.rgb" \
            -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l)
        echo "[$(date +%H:%M:%S)] DONE  $scene ($n_rgb frames)" | tee -a "$SUMMARY"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $scene (see ${LOG_DIR}/${scene}.log)" | tee -a "$SUMMARY"
    fi
done < <(find "$TRAJ_DIR" -maxdepth 1 -type f -name '*.tar.gz' | sort)

echo "Batch complete: $(date)" | tee -a "$SUMMARY"
