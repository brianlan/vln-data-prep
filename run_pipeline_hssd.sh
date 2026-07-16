#!/bin/bash
set -euo pipefail

set -a
export http_proxy=http://127.0.0.1:18080
export https_proxy=http://127.0.0.1:18080
export HTTP_PROXY=http://127.0.0.1:18080
export HTTPS_PROXY=http://127.0.0.1:18080
export PATH="/home/rlan/.local/bin:$PATH"
export LD_LIBRARY_PATH="/home/rlan/.local/bin:$LD_LIBRARY_PATH"
set +a

PYTHON=/ssd4/envs/vln_py311/bin/python
BLENDERPROC=/ssd4/envs/vln_py311/bin/blenderproc
SCRIPT_DIR=/home/rlan/projects/vln-data-prep

HSSD_ROOT=/ssd5/datasets/hssd-hab
TRAJ_DIR=/ssd5/datasets/InternData-N1/vln_n1/traj_data/hssd_zed
OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/hssd
WORK_DIR=/tmp/opencode/hssd_fisheye_work
GLB_CACHE="${WORK_DIR}/glbs_decompressed"

WIDTH=640
HEIGHT=640
FOV_DEG=195.0

SCENE=${1:-102344049}

echo "============================================"
echo "  HSSD Fisheye Rendering Pipeline"
echo "  Scene: $SCENE"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  FOV: ${FOV_DEG}°"
echo "============================================"

TRAJ_TAR="${TRAJ_DIR}/${SCENE}.tar.gz"
EXTRACTED_DIR="${WORK_DIR}/${SCENE}"
RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered"
SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"
SCENE_INSTANCE="${HSSD_ROOT}/scenes/${SCENE}.scene_instance.json"

if [ ! -f "$TRAJ_TAR" ]; then
    echo "ERROR: Trajectory tar not found: $TRAJ_TAR"
    exit 1
fi
if [ ! -f "$SCENE_INSTANCE" ]; then
    echo "ERROR: Scene instance not found: $SCENE_INSTANCE"
    exit 1
fi

echo ""
echo "[1/5] Extracting trajectory data..."
rm -rf "$EXTRACTED_DIR"
mkdir -p "$EXTRACTED_DIR"
tar -xzf "$TRAJ_TAR" -C "$EXTRACTED_DIR"
EXTRACTED_TRAJ="${EXTRACTED_DIR}/${SCENE}"
echo "  Extracted to: $EXTRACTED_TRAJ"

echo ""
echo "[2/5] Pre-extracting trajectory poses..."
TRAJ_NPY_DIR="${WORK_DIR}/${SCENE}_npy"
rm -rf "$TRAJ_NPY_DIR"
mkdir -p "$TRAJ_NPY_DIR"
$PYTHON "${SCRIPT_DIR}/prepare_trajectories.py" \
    --traj_dir "$EXTRACTED_TRAJ" \
    --output_dir "$TRAJ_NPY_DIR"

echo ""
echo "[3/5] Decompressing KTX2/BasisU GLB textures..."
if [ ! -d "$GLB_CACHE" ] || [ "$(ls -A "$GLB_CACHE" 2>/dev/null | wc -l)" -eq 0 ]; then
    $PYTHON "${SCRIPT_DIR}/decompress_glbs.py" \
        --scene_instance "$SCENE_INSTANCE" \
        --hssd_root "$HSSD_ROOT" \
        --output_dir "$GLB_CACHE"
else
    echo "  GLB cache already exists: $GLB_CACHE ($(ls "$GLB_CACHE" | wc -l) files)"
fi

echo ""
echo "[4/5] Rendering fisheye images..."
rm -rf "$RENDERED_DIR"
mkdir -p "$RENDERED_DIR"
$BLENDERPROC run "${SCRIPT_DIR}/render_fisheye_hssd.py" \
    --scene "$SCENE" \
    --scene_instance "$SCENE_INSTANCE" \
    --hssd_root "$HSSD_ROOT" \
    --glb_cache "$GLB_CACHE" \
    --traj_dir "$TRAJ_NPY_DIR" \
    --output_dir "$RENDERED_DIR" \
    --width $WIDTH \
    --height $HEIGHT \
    --fov_deg $FOV_DEG

echo ""
echo "[5/5] Packaging into LeRobot v2.1 format..."
rm -rf "$SCENE_OUTPUT"
mkdir -p "$SCENE_OUTPUT"
$PYTHON "${SCRIPT_DIR}/package_lerobot.py" \
    --scene "$SCENE" \
    --traj_dir "$EXTRACTED_TRAJ" \
    --rendered_dir "$RENDERED_DIR" \
    --output_dir "$SCENE_OUTPUT" \
    --width $WIDTH \
    --height $HEIGHT \
    --fov_deg $FOV_DEG

echo ""
echo "============================================"
echo "  DONE! Scene $SCENE processed."
echo "  Output: $SCENE_OUTPUT"
echo "============================================"
