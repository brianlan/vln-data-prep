#!/bin/bash
set -euo pipefail

# Render one Scene-N1 MP3D scene with InternData-N1 matterport3d_zed poses.
# Sparse smoke tests intentionally skip LeRobot packaging.

PYTHON=/ssd4/envs/vln_data_prep_py311/bin/python
BLENDERPROC=/ssd4/envs/vln_data_prep_py311/bin/blenderproc
SCRIPT_DIR=/home/rlan/projects/vln-data-prep

MP3D_ROOT=/ssd5/datasets/Scene-N1/mp3d_n1
TRAJ_DIR=/ssd5/datasets/InternData-N1/vln_n1/traj_data/matterport3d_zed
OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/mp3d
SMOKE_OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/mp3d-smoke
WORK_DIR=/tmp/opencode/mp3d_fisheye_work

WIDTH=640
HEIGHT=640
FOV_DEG=195.0
SAMPLES=16
MAX_EPISODES=0
FRAME_STRIDE=1
SMOKE_TEST=0

usage() {
    echo "Usage: bash run_pipeline_mp3d.sh [scene] [--smoke-test] [--samples N]"
}

SCENE=29hnd4uzFmX
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

SCENE_ASSET_ROOT="${MP3D_ROOT}/${SCENE}/matterport_mesh"
OBJ_CANDIDATES=()
while IFS= read -r obj; do
    OBJ_CANDIDATES+=("$obj")
done < <(find "$SCENE_ASSET_ROOT" -mindepth 2 -maxdepth 2 -type f -name '*.obj' 2>/dev/null | sort)
if [[ ${#OBJ_CANDIDATES[@]} -ne 1 ]]; then
    echo "ERROR: Expected exactly one MP3D render OBJ for ${SCENE}, found ${#OBJ_CANDIDATES[@]}."
    echo "Searched: ${SCENE_ASSET_ROOT}/*/*.obj"
    exit 1
fi
MESH_PATH=${OBJ_CANDIDATES[0]}
MTL_PATH=${MESH_PATH%.obj}.mtl
MESH_DIR=$(dirname "$MESH_PATH")

if [[ ! -f "$MTL_PATH" ]]; then
    echo "ERROR: MP3D material file not found: $MTL_PATH"
    exit 1
fi
if ! find "$MESH_DIR" -maxdepth 1 -type f -name '*.jpg' -print -quit | grep -q .; then
    echo "ERROR: No MP3D JPEG texture tiles found beside: $MESH_PATH"
    exit 1
fi

TRAJ_TAR="${TRAJ_DIR}/${SCENE}.tar.gz"
EXTRACTED_ROOT="${WORK_DIR}/${SCENE}"
EXTRACTED_TRAJ="${EXTRACTED_ROOT}/${SCENE}"
TRAJ_NPY_DIR="${WORK_DIR}/${SCENE}_npy"

if [[ $SMOKE_TEST -eq 1 ]]; then
    RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered_smoke"
    SCENE_OUTPUT="${SMOKE_OUTPUT_ROOT}/${SCENE}"
else
    RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered"
    SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"
fi

echo "MP3D fisheye pipeline: scene=${SCENE}, mesh=${MESH_PATH}, smoke_test=${SMOKE_TEST}"

if [[ ! -f "$TRAJ_TAR" ]]; then
    echo "ERROR: Trajectory archive not found: $TRAJ_TAR"
    exit 1
fi
if head -n 1 "$TRAJ_TAR" | grep -q 'git-lfs.github.com/spec'; then
    echo "ERROR: $TRAJ_TAR is only a Git LFS pointer."
    echo "Fetch it with:"
    echo "  cd /ssd5/datasets/InternData-N1"
    echo "  git lfs pull --include=\"vln_n1/traj_data/matterport3d_zed/${SCENE}.tar.gz\" --exclude=\"\""
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
"$BLENDERPROC" run "${SCRIPT_DIR}/render_fisheye_mp3d.py" \
    --scene "$SCENE" \
    --mesh "$MESH_PATH" \
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
