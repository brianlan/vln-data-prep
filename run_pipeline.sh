#!/bin/bash
set -euo pipefail

# ============================================================
# Pipeline: Render fisheye images from Replica scenes + trajectories
# Produces LeRobot v2.1 format dataset at /ssd5/datasets/vln-fisheye/replica/
# ============================================================

set -a
export http_proxy=http://127.0.0.1:18080
export https_proxy=http://127.0.0.1:18080
export HTTP_PROXY=http://127.0.0.1:18080
export HTTPS_PROXY=http://127.0.0.1:18080
set +a

PYTHON=/ssd4/envs/vln_py311/bin/python
BLENDERPROC=/ssd4/envs/vln_py311/bin/blenderproc
SCRIPT_DIR=/home/rlan/projects/vln-data-prep

REPLICA_DIR=/ssd5/datasets/Replica
TRAJ_DIR=/ssd5/datasets/InternData-N1/vln_n1/traj_data/replica_zed
OUTPUT_ROOT=/ssd5/datasets/vln-fisheye/replica
WORK_DIR=/tmp/opencode/replica_fisheye_work

# Fisheye params
WIDTH=640
HEIGHT=640
FOV_DEG=195.0

# Scene to process (pass as argument, default: apartment_1)
SCENE=${1:-apartment_1}

echo "============================================"
echo "  Fisheye Rendering Pipeline"
echo "  Scene: $SCENE"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  FOV: ${FOV_DEG}°"
echo "============================================"

# --- 1. Extract original trajectory tar.gz ---
TRAJ_TAR="${TRAJ_DIR}/${SCENE}.tar.gz"
EXTRACTED_DIR="${WORK_DIR}/${SCENE}"
RENDERED_DIR="${WORK_DIR}/${SCENE}_rendered"
SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"

if [ ! -f "$TRAJ_TAR" ]; then
    echo "ERROR: Trajectory tar not found: $TRAJ_TAR"
    exit 1
fi

echo ""
echo "[1/4] Extracting trajectory data..."
rm -rf "$EXTRACTED_DIR"
mkdir -p "$EXTRACTED_DIR"
tar -xzf "$TRAJ_TAR" -C "$EXTRACTED_DIR"
# The tar extracts to $EXTRACTED_DIR/$SCENE/
EXTRACTED_TRAJ="${EXTRACTED_DIR}/${SCENE}"
echo "  Extracted to: $EXTRACTED_TRAJ"

# --- 2. Pre-extract trajectory poses to .npy ---
echo ""
echo "[2/4] Pre-extracting trajectory poses to .npy..."
TRAJ_NPY_DIR="${WORK_DIR}/${SCENE}_npy"
rm -rf "$TRAJ_NPY_DIR"
mkdir -p "$TRAJ_NPY_DIR"

$PYTHON "${SCRIPT_DIR}/prepare_trajectories.py" \
    --traj_dir "$EXTRACTED_TRAJ" \
    --output_dir "$TRAJ_NPY_DIR"

# --- 3. Render fisheye images ---
MESH_PATH="${REPLICA_DIR}/${SCENE}/mesh.ply"
if [ ! -f "$MESH_PATH" ]; then
    echo "ERROR: Replica mesh not found: $MESH_PATH"
    exit 1
fi

echo ""
echo "[3/4] Rendering fisheye images..."
rm -rf "$RENDERED_DIR"
mkdir -p "$RENDERED_DIR"

$BLENDERPROC run "${SCRIPT_DIR}/render_fisheye.py" \
    --scene "$SCENE" \
    --mesh "$MESH_PATH" \
    --traj_dir "$TRAJ_NPY_DIR" \
    --output_dir "$RENDERED_DIR" \
    --width $WIDTH \
    --height $HEIGHT \
    --fov_deg $FOV_DEG

# --- 4. Package into LeRobot v2.1 format ---
echo ""
echo "[4/4] Packaging into LeRobot v2.1 format..."
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
