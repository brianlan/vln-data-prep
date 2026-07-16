#!/bin/bash
set -euo pipefail

# Render one HM3D scene with InternData-N1 hm3d_zed trajectories. A smoke test
# renders sparse frames and deliberately skips LeRobot packaging.

PYTHON=/ssd4/envs/vln_data_prep_py311/bin/python
BLENDERPROC=/ssd4/envs/vln_data_prep_py311/bin/blenderproc
SCRIPT_DIR=/home/rlan/projects/vln-data-prep

HM3D_ROOT=/ssd5/datasets/hm3d/versioned_data/hm3d-0.2/hm3d
TRAJ_DIR=/ssd5/datasets/InternData-N1/vln_n1/traj_data/hm3d_zed
OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/hm3d
SMOKE_OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/hm3d-smoke
WORK_DIR=/tmp/opencode/hm3d_fisheye_work

WIDTH=640
HEIGHT=640
FOV_DEG=195.0
SAMPLES=16
MAX_EPISODES=0
FRAME_STRIDE=1
SMOKE_TEST=0

usage() {
    echo "Usage: bash run_pipeline_hm3d.sh [scene] [--smoke-test] [--samples N]"
}

SCENE=00275-4dbCzNN5L5t
if [[ $# -gt 0 && "$1" != --* ]]; then
    SCENE=$1
    shift
fi

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

GLB_CANDIDATES=()
for split in train val; do
    while IFS= read -r glb; do
        GLB_CANDIDATES+=("$glb")
    done < <(find "${HM3D_ROOT}/${split}/${SCENE}" -maxdepth 1 -type f -name '*.glb' 2>/dev/null | sort)
done
if [[ ${#GLB_CANDIDATES[@]} -ne 1 ]]; then
    echo "ERROR: Expected exactly one HM3D GLB for ${SCENE}, found ${#GLB_CANDIDATES[@]}."
    echo "Searched: ${HM3D_ROOT}/{train,val}/${SCENE}/*.glb"
    exit 1
fi
GLB_PATH=${GLB_CANDIDATES[0]}

if [[ $SMOKE_TEST -eq 1 ]]; then
    RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered_smoke"
    SCENE_OUTPUT="${SMOKE_OUTPUT_ROOT}/${SCENE}"
else
    RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered"
    SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"
fi

echo "HM3D fisheye pipeline: scene=${SCENE}, glb=${GLB_PATH}, smoke_test=${SMOKE_TEST}"

if [[ ! -f "$TRAJ_TAR" ]]; then
    echo "ERROR: Trajectory archive not found: $TRAJ_TAR"
    exit 1
fi
if head -n 1 "$TRAJ_TAR" | grep -q 'git-lfs.github.com/spec'; then
    echo "ERROR: $TRAJ_TAR is only a Git LFS pointer."
    echo "Fetch it with:"
    echo "  cd /ssd5/datasets/InternData-N1"
    echo "  git lfs pull --include=\"vln_n1/traj_data/hm3d_zed/${SCENE}.tar.gz\" --exclude=\"\""
    exit 1
fi
if ! gzip -t "$TRAJ_TAR"; then
    echo "ERROR: Trajectory archive failed gzip integrity check: $TRAJ_TAR"
    exit 1
fi

echo "[1/4] Extracting trajectory archive"
rm -rf "$EXTRACTED_ROOT"
mkdir -p "$EXTRACTED_ROOT"
tar -xzf "$TRAJ_TAR" -C "$EXTRACTED_ROOT"
if [[ ! -d "$EXTRACTED_TRAJ" ]]; then
    echo "ERROR: Expected extracted trajectory directory not found: $EXTRACTED_TRAJ"
    exit 1
fi

echo "[2/4] Preparing trajectory poses"
rm -rf "$TRAJ_NPY_DIR"
mkdir -p "$TRAJ_NPY_DIR"
"$PYTHON" "${SCRIPT_DIR}/prepare_trajectories.py" \
    --traj_dir "$EXTRACTED_TRAJ" \
    --output_dir "$TRAJ_NPY_DIR"

echo "[3/4] Rendering textured fisheye RGB/depth"
rm -rf "$RENDERED_DIR" "$SCENE_OUTPUT"
mkdir -p "$RENDERED_DIR" "$SCENE_OUTPUT"
"$BLENDERPROC" run "${SCRIPT_DIR}/render_fisheye_hm3d.py" \
    --scene "$SCENE" \
    --glb "$GLB_PATH" \
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
    echo "[4/4] Smoke test: skipped LeRobot packaging because frames are sparse"
    echo "DONE: $SCENE_OUTPUT"
    exit 0
fi

echo "[4/4] Packaging complete scene as LeRobot v2.1"
"$PYTHON" "${SCRIPT_DIR}/package_lerobot.py" \
    --scene "$SCENE" \
    --traj_dir "$EXTRACTED_TRAJ" \
    --rendered_dir "$RENDERED_DIR" \
    --output_dir "$SCENE_OUTPUT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fov_deg "$FOV_DEG"

echo "DONE: $SCENE_OUTPUT"
