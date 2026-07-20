#!/bin/bash
set -euo pipefail

# Generate PointGoal trajectories, render native SAGE3D fisheye RGB/depth, and
# package a complete LeRobot v2.1 scene.

ISAAC_PYTHON=/ssd4/envs/isaac_sim_py311/bin/python
PACKAGE_PYTHON=/ssd4/envs/vln_data_prep_py311/bin/python
SCRIPT_DIR=/home/rlan/projects/vln-data-prep

SAGE_ROOT=/ssd5/datasets/SAGE3D
OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/sage3d
WORK_ROOT=/tmp/opencode/sage3d_pointgoal

EPISODES=5
SEED=20260720
ROBOT_RADIUS=0.25
SAFETY_MARGIN=0.05
CAMERA_HEIGHT=0.6
MIN_PATH_LENGTH=3.0
MAX_PATH_LENGTH=15.0
FRAME_SPACING=0.05
WIDTH=640
HEIGHT=640
FOV_DEG=195.0
FORCE=0
PLAN_ONLY=0

usage() {
    echo "Usage: bash run_pipeline_sage3d.sh <scene-id> [options]"
    echo "Options:"
    echo "  --episodes N"
    echo "  --seed N"
    echo "  --plan-only"
    echo "  --force"
}

if [[ $# -eq 0 || "$1" == --* ]]; then
    usage
    exit 2
fi
SCENE=$1
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            EPISODES=$2
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            SEED=$2
            shift 2
            ;;
        --plan-only)
            PLAN_ONLY=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

USDZ="${SAGE_ROOT}/InteriorGS_usdz/${SCENE}.usdz"
COLLISION_USD="${SAGE_ROOT}/Collision_Mesh/Collision_Mesh/${SCENE}/${SCENE}_collision.usd"
WORK_DIR="${WORK_ROOT}/${SCENE}"
TRAJECTORY_DIR="${WORK_DIR}/trajectories"
RENDERED_DIR="${WORK_DIR}/rendered"
SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"

for path in "$USDZ" "$COLLISION_USD"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: Required SAGE3D asset not found: $path"
        exit 1
    fi
done

if [[ -e "$SCENE_OUTPUT" && $FORCE -ne 1 && $PLAN_ONLY -ne 1 ]]; then
    echo "ERROR: Output already exists: $SCENE_OUTPUT"
    echo "Use --force to replace this generated scene."
    exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$TRAJECTORY_DIR"

echo "[1/4] Generating safe PointGoal trajectories for ${SCENE}"
"$ISAAC_PYTHON" "${SCRIPT_DIR}/generate_sage3d_trajectories.py" \
    --scene "$SCENE" \
    --interiorgs-root "${SAGE_ROOT}/InteriorGS" \
    --collision-usd "$COLLISION_USD" \
    --output-dir "$TRAJECTORY_DIR" \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --robot-radius "$ROBOT_RADIUS" \
    --safety-margin "$SAFETY_MARGIN" \
    --camera-height "$CAMERA_HEIGHT" \
    --min-path-length "$MIN_PATH_LENGTH" \
    --max-path-length "$MAX_PATH_LENGTH" \
    --frame-spacing "$FRAME_SPACING"

if [[ $PLAN_ONLY -eq 1 ]]; then
    echo "DONE (plan only): $TRAJECTORY_DIR"
    exit 0
fi

echo "[2/4] Rendering 3DGS RGB and collision-mesh ray depth"
mkdir -p "$RENDERED_DIR"
"$ISAAC_PYTHON" "${SCRIPT_DIR}/render_fisheye_sage3d.py" \
    --mode rgb \
    --scene "$SCENE" \
    --usdz "$USDZ" \
    --collision-usd "$COLLISION_USD" \
    --trajectory-dir "$TRAJECTORY_DIR" \
    --output-dir "$RENDERED_DIR" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fov-deg "$FOV_DEG"
"$ISAAC_PYTHON" "${SCRIPT_DIR}/render_fisheye_sage3d.py" \
    --mode depth \
    --scene "$SCENE" \
    --usdz "$USDZ" \
    --collision-usd "$COLLISION_USD" \
    --trajectory-dir "$TRAJECTORY_DIR" \
    --output-dir "$RENDERED_DIR" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fov-deg "$FOV_DEG"

echo "[3/4] Packaging LeRobot v2.1 PointGoal dataset"
rm -rf "$SCENE_OUTPUT"
mkdir -p "$SCENE_OUTPUT"
"$PACKAGE_PYTHON" "${SCRIPT_DIR}/package_lerobot_sage3d.py" \
    --scene "$SCENE" \
    --trajectory-dir "$TRAJECTORY_DIR" \
    --rendered-dir "$RENDERED_DIR" \
    --output-dir "$SCENE_OUTPUT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fov-deg "$FOV_DEG" \
    --camera-height "$CAMERA_HEIGHT"

echo "[4/4] Verifying output inventory"
EXPECTED_FRAMES=$(
    jq '[.episodes[].frame_count] | add' \
        "${TRAJECTORY_DIR}/trajectory_manifest.json"
)
RGB_FRAMES=$(
    find "${SCENE_OUTPUT}/videos/chunk-000/observation.images.rgb" \
        -maxdepth 1 -type f -name '*.jpg' | wc -l
)
DEPTH_FRAMES=$(
    find "${SCENE_OUTPUT}/videos/chunk-000/observation.images.depth" \
        -maxdepth 1 -type f -name '*.png' | wc -l
)
PARQUETS=$(
    find "${SCENE_OUTPUT}/data/chunk-000" \
        -maxdepth 1 -type f -name '*.parquet' | wc -l
)

if [[ "$RGB_FRAMES" -ne "$EXPECTED_FRAMES" ||
      "$DEPTH_FRAMES" -ne "$EXPECTED_FRAMES" ||
      "$PARQUETS" -ne "$EPISODES" ]]; then
    echo "ERROR: Output inventory mismatch:"
    echo "  expected_frames=${EXPECTED_FRAMES}"
    echo "  rgb=${RGB_FRAMES}, depth=${DEPTH_FRAMES}, parquet=${PARQUETS}"
    exit 1
fi

echo "DONE: ${SCENE_OUTPUT}"
echo "  episodes=${EPISODES}, frames=${EXPECTED_FRAMES}"
