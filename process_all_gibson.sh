#!/bin/bash
set -uo pipefail

# Process every real Gibson Zed trajectory whose V2 scene assets are available.
# LFS pointer archives are reported and skipped by run_pipeline_gibson.sh.

SCRIPT_DIR=/home/rlan/projects/vln-data-prep
TRAJ_DIR=/ssd5/datasets/InternData-N1/vln_n1/traj_data/gibson_zed
LOG_DIR=/tmp/opencode/gibson_batch_logs
mkdir -p "$LOG_DIR"

SUMMARY="${LOG_DIR}/batch_summary.log"
echo "Batch start: $(date)" | tee "$SUMMARY"

while IFS= read -r archive; do
    scene=$(basename "$archive" .tar.gz)
    echo "[$(date +%H:%M:%S)] START $scene" | tee -a "$SUMMARY"
    if bash "${SCRIPT_DIR}/run_pipeline_gibson.sh" "$scene" > "${LOG_DIR}/${scene}.log" 2>&1; then
        n_rgb=$(find "/ssd5/datasets/vln-fisheye/gibson/${scene}/videos/chunk-000/observation.images.rgb" \
            -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | wc -l)
        echo "[$(date +%H:%M:%S)] DONE  $scene ($n_rgb frames)" | tee -a "$SUMMARY"
    else
        echo "[$(date +%H:%M:%S)] SKIP/FAIL $scene (see ${LOG_DIR}/${scene}.log)" | tee -a "$SUMMARY"
    fi
done < <(find "$TRAJ_DIR" -maxdepth 1 -type f -name '*.tar.gz' | sort)

echo "Batch complete: $(date)" | tee -a "$SUMMARY"
