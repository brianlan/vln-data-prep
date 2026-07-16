#!/bin/bash
# Batch process remaining Replica scenes
set -uo pipefail

SCRIPT_DIR=/home/rlan/projects/vln-data-prep
LOG_DIR=/tmp/opencode/replica_batch_logs
mkdir -p "$LOG_DIR"

SCENES=(
    apartment_0
    frl_apartment_0
    frl_apartment_1
    frl_apartment_2
    frl_apartment_3
    frl_apartment_4
    frl_apartment_5
    office_0
    office_4
    room_0
)

echo "Batch start: $(date)" | tee "$LOG_DIR/batch_summary.log"

for scene in "${SCENES[@]}"; do
    echo "========================================" | tee -a "$LOG_DIR/batch_summary.log"
    echo "[$(date +%H:%M:%S)] START $scene" | tee -a "$LOG_DIR/batch_summary.log"

    if bash "$SCRIPT_DIR/run_pipeline.sh" "$scene" > "$LOG_DIR/${scene}.log" 2>&1; then
        n_rgb=$(ls /ssd5/datasets/vln-fisheye/replica/$scene/videos/chunk-000/observation.images.rgb/*.jpg 2>/dev/null | wc -l)
        echo "[$(date +%H:%M:%S)] DONE  $scene ($n_rgb frames)" | tee -a "$LOG_DIR/batch_summary.log"
    else
        echo "[$(date +%H:%M:%S)] FAIL  $scene (see ${scene}.log)" | tee -a "$LOG_DIR/batch_summary.log"
    fi
done

echo "========================================" | tee -a "$LOG_DIR/batch_summary.log"
echo "Batch complete: $(date)" | tee -a "$LOG_DIR/batch_summary.log"
