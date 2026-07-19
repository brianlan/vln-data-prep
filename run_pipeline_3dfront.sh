#!/bin/bash
set -euo pipefail

# Render one official 3D-FRONT scene with InternData-N1 3dfront_zed poses.
# Sparse smoke tests intentionally skip LeRobot packaging.

PYTHON=/ssd4/envs/vln_data_prep_py311/bin/python
BLENDERPROC=/ssd4/envs/vln_data_prep_py311/bin/blenderproc
SCRIPT_DIR=/home/rlan/projects/vln-data-prep

FRONT_ROOT=/ssd5/datasets/3dfront
ASSET_CACHE_ROOT=/ssd5/datasets/3dfront/prepared
TRAJ_DIR=/ssd5/datasets/InternData-N1/vln_n1/traj_data/3dfront_zed
OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/3dfront
SMOKE_OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/3dfront-smoke
WORK_DIR=/tmp/opencode/3dfront_fisheye_work

WIDTH=640
HEIGHT=640
FOV_DEG=195.0
SAMPLES=16
MAX_EPISODES=0
FRAME_STRIDE=1
SMOKE_TEST=0

usage() {
    echo "Usage: bash run_pipeline_3dfront.sh <scene> [--smoke-test] [--samples N]"
}

if [[ $# -eq 0 || "$1" == --* ]]; then
    usage
    exit 2
fi
SCENE=$1
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            SMOKE_TEST=1
            MAX_EPISODES=1
            FRAME_STRIDE=20
            shift
            ;;
        --samples)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            SAMPLES=$2
            shift 2
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

TRAJ_TAR="${TRAJ_DIR}/${SCENE}.tar.gz"
EXTRACTED_ROOT="${WORK_DIR}/${SCENE}"
EXTRACTED_TRAJ="${EXTRACTED_ROOT}/${SCENE}"
TRAJ_NPY_DIR="${WORK_DIR}/${SCENE}_npy"
ASSET_ROOT="${ASSET_CACHE_ROOT}/${SCENE}"
SCENE_JSON="${ASSET_ROOT}/${SCENE}.json"

if [[ $SMOKE_TEST -eq 1 ]]; then
    RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered_smoke"
    SCENE_OUTPUT="${SMOKE_OUTPUT_ROOT}/${SCENE}"
else
    RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered"
    SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"
fi

echo "3D-FRONT fisheye pipeline: scene=${SCENE}, smoke_test=${SMOKE_TEST}"

for archive in \
    3D-FRONT.zip \
    3D-FRONT-texture.zip \
    3D-FUTURE-model-part1.zip \
    3D-FUTURE-model-part2.zip \
    3D-FUTURE-model-part3.zip \
    3D-FUTURE-model-part4.zip; do
    if [[ ! -f "${FRONT_ROOT}/${archive}" ]]; then
        echo "ERROR: Required 3D-FRONT archive not found: ${FRONT_ROOT}/${archive}"
        exit 1
    fi
done

if [[ ! -f "$TRAJ_TAR" ]]; then
    echo "ERROR: Trajectory archive not found: $TRAJ_TAR"
    exit 1
fi
if head -n 1 "$TRAJ_TAR" | grep -q 'git-lfs.github.com/spec'; then
    echo "ERROR: $TRAJ_TAR is only a Git LFS pointer."
    echo "Fetch it with:"
    echo "  cd /ssd5/datasets/InternData-N1"
    echo "  git lfs pull --include=\"vln_n1/traj_data/3dfront_zed/${SCENE}.tar.gz\" --exclude=\"\""
    exit 1
fi
if ! gzip -t "$TRAJ_TAR"; then
    echo "ERROR: Trajectory archive failed gzip integrity check: $TRAJ_TAR"
    exit 1
fi

echo "[1/5] Preparing the scene's JSON, furniture, and textures"
"$PYTHON" "${SCRIPT_DIR}/prepare_3dfront_assets.py" \
    --scene "$SCENE" \
    --dataset_root "$FRONT_ROOT" \
    --output_root "$ASSET_CACHE_ROOT"

echo "[2/5] Extracting trajectory archive"
rm -rf "$EXTRACTED_ROOT"
mkdir -p "$EXTRACTED_ROOT"
tar -xzf "$TRAJ_TAR" -C "$EXTRACTED_ROOT"
if [[ ! -d "$EXTRACTED_TRAJ" ]]; then
    echo "ERROR: Expected extracted trajectory directory not found: $EXTRACTED_TRAJ"
    exit 1
fi

echo "[3/5] Preparing trajectory poses"
rm -rf "$TRAJ_NPY_DIR"
mkdir -p "$TRAJ_NPY_DIR"
"$PYTHON" "${SCRIPT_DIR}/prepare_trajectories.py" \
    --traj_dir "$EXTRACTED_TRAJ" \
    --output_dir "$TRAJ_NPY_DIR"

echo "[4/5] Rendering 3D-FRONT fisheye RGB/depth"
rm -rf "$RENDERED_DIR" "$SCENE_OUTPUT"
mkdir -p "$RENDERED_DIR" "$SCENE_OUTPUT"
"$BLENDERPROC" run "${SCRIPT_DIR}/render_fisheye_3dfront.py" \
    --scene "$SCENE" \
    --scene_json "$SCENE_JSON" \
    --future_model_path "${ASSET_ROOT}/3D-FUTURE-model" \
    --front_texture_path "${ASSET_ROOT}/3D-FRONT-texture" \
    --traj_dir "$TRAJ_NPY_DIR" \
    --output_dir "$RENDERED_DIR" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fov_deg "$FOV_DEG" \
    --samples "$SAMPLES" \
    --max_episodes "$MAX_EPISODES" \
    --frame_stride "$FRAME_STRIDE"

if [[ $SMOKE_TEST -eq 1 ]]; then
    cp -a "$RENDERED_DIR"/. "$SCENE_OUTPUT"/
    echo "[5/5] Smoke test: skipped LeRobot packaging because frames are sparse"
    echo "DONE: $SCENE_OUTPUT"
    exit 0
fi

echo "[5/5] Packaging complete scene as LeRobot v2.1"
"$PYTHON" "${SCRIPT_DIR}/package_lerobot.py" \
    --scene "$SCENE" \
    --traj_dir "$EXTRACTED_TRAJ" \
    --rendered_dir "$RENDERED_DIR" \
    --output_dir "$SCENE_OUTPUT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fov_deg "$FOV_DEG"

echo "DONE: $SCENE_OUTPUT"
