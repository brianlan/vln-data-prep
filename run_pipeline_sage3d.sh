#!/bin/bash
set -euo pipefail

# Generate PointGoal trajectories, render native SAGE3D fisheye RGB/depth, and
# package a complete LeRobot v2.1 scene.
#
# Phase 6 portability/rollout contract:
# - Interpreters are env-selected (SAGE3D_ISAAC_PYTHON / SAGE3D_PACKAGE_PYTHON)
#   with the documented local defaults below; override them on other machines.
# - OUTPUT_ROOT is operator-owned and must already exist as a real directory.
# - WORK_ROOT is disposable and is created with guarded single-component mkdir.
# - Producers (generate/package) allocate their own sibling staging stages and
#   never pre-create their final targets; the shell never pre-creates
#   publication targets.
# - The shared render stage is allocated by the package-safe allocator
#   (sage3d.cli.create_staging), rendered rgb then depth into that stage by
#   sage3d.cli.render, and published by the package-Python finalizer
#   (sage3d.cli.finalize_render).
# - Destructive cleanup is confined to explicit shell-owned operations
#   (--force removal and the disposable work-directory reset) and is protected
#   by numeric-scene / canonical-root / strict-descendant / symlink guardrails.

ISAAC_PYTHON="${SAGE3D_ISAAC_PYTHON:-/ssd4/envs/isaac_sim_py311/bin/python}"
PACKAGE_PYTHON="${SAGE3D_PACKAGE_PYTHON:-/ssd4/envs/vln_data_prep_py311/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
WIDTH=600
HEIGHT=450
HORIZONTAL_FOV_DEG=180.0
FISHEYE_COEFFICIENTS=(0.1 0.0 0.0 0.0)
FORCE=0
PLAN_ONLY=0

usage() {
    echo "Usage: bash run_pipeline_sage3d.sh <scene-id> [options]"
    echo "Options:"
    echo "  --episodes N"
    echo "  --seed N"
    echo "  --width N"
    echo "  --height N"
    echo "  --horizontal-fov-deg DEGREES"
    echo "  --fisheye-coefficients K1 K2 K3 K4"
    echo "  --output-root PATH"
    echo "  --work-root PATH"
    echo "  --plan-only"
    echo "  --force"
    echo "Environment:"
    echo "  SAGE3D_ISAAC_PYTHON    Isaac Sim python (default: /ssd4/envs/isaac_sim_py311/bin/python)"
    echo "  SAGE3D_PACKAGE_PYTHON  package-safe python (default: /ssd4/envs/vln_data_prep_py311/bin/python)"
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
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
        --width)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            WIDTH=$2
            shift 2
            ;;
        --height)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            HEIGHT=$2
            shift 2
            ;;
        --horizontal-fov-deg)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            HORIZONTAL_FOV_DEG=$2
            shift 2
            ;;
        --fisheye-coefficients)
            [[ $# -ge 5 ]] || { usage; exit 2; }
            FISHEYE_COEFFICIENTS=("$2" "$3" "$4" "$5")
            shift 5
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            OUTPUT_ROOT=$2
            shift 2
            ;;
        --work-root)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            WORK_ROOT=$2
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

# --- destructive-shell guardrails -------------------------------------------

# Scene IDs must be numeric so destructive targets are always single-component
# descendants of a declared root.
[[ "$SCENE" =~ ^[0-9]+$ ]] || fail "scene ID must be numeric: $SCENE"

# OUTPUT_ROOT is durable/operator-owned: must already be a nonempty path naming
# an existing real directory (may be empty) and must not be a symlink.
[[ -n "$OUTPUT_ROOT" && "$OUTPUT_ROOT" != "/" ]] \
    || fail "OUTPUT_ROOT must be a nonempty non-root path"
[[ -e "$OUTPUT_ROOT" ]] || fail "OUTPUT_ROOT must already exist: $OUTPUT_ROOT"
[[ -L "$OUTPUT_ROOT" ]] && fail "OUTPUT_ROOT must not be a symlink: $OUTPUT_ROOT"
[[ -d "$OUTPUT_ROOT" ]] || fail "OUTPUT_ROOT must be a real directory: $OUTPUT_ROOT"

# WORK_ROOT is disposable. When absent, create exactly one directory with plain
# mkdir (never mkdir -p) after validating the single-component basename and the
# real-directory parent; when present, require the same real-directory checks.
[[ -n "$WORK_ROOT" && "$WORK_ROOT" != "/" ]] \
    || fail "WORK_ROOT must be a nonempty non-root path"
_WORK_BASE="$(basename -- "$WORK_ROOT")"
_WORK_PARENT="$(dirname -- "$WORK_ROOT")"
[[ -n "$_WORK_BASE" && "$_WORK_BASE" != "." && "$_WORK_BASE" != ".." ]] \
    || fail "WORK_ROOT basename must be one component: $WORK_ROOT"
if [[ -e "$WORK_ROOT" ]]; then
    [[ -L "$WORK_ROOT" ]] && fail "WORK_ROOT must not be a symlink: $WORK_ROOT"
    [[ -d "$WORK_ROOT" ]] || fail "WORK_ROOT must be a real directory: $WORK_ROOT"
else
    [[ -d "$_WORK_PARENT" && ! -L "$_WORK_PARENT" ]] \
        || fail "WORK_ROOT parent must be an existing real directory: $_WORK_PARENT"
    realpath -e "$_WORK_PARENT" >/dev/null \
        || fail "WORK_ROOT parent must resolve: $_WORK_PARENT"
    mkdir "$WORK_ROOT"
    [[ -d "$WORK_ROOT" && ! -L "$WORK_ROOT" ]] \
        || fail "WORK_ROOT creation failed: $WORK_ROOT"
fi

# Validate a destructive target before any rm -rf. Resolves the existing parent
# with realpath -e, rejects symlinked roots/targets, forms the candidate from
# the resolved parent plus the validated single-component scene name, requires
# a strict descendant of the declared root, refuses empty/root/self paths and
# .. traversal, and refuses anything equal to or nested under SCRIPT_DIR.
# Prints the exact validated target on success.
guard_destructive_target() {
    local target="$1" root="$2" label="$3"
    local parent base resolved_parent candidate resolved_root
    [[ -n "$target" && "$target" != "/" ]] \
        || fail "$label must be a non-root path: $target"
    parent="$(dirname -- "$target")"
    base="$(basename -- "$target")"
    [[ -n "$base" && "$base" != "." && "$base" != ".." ]] \
        || fail "$label basename must be one component: $target"
    [[ -e "$parent" && ! -L "$parent" ]] \
        || fail "$label parent must be an existing real directory: $parent"
    resolved_parent="$(realpath -e "$parent")" \
        || fail "$label parent must resolve: $parent"
    if [[ -e "$target" ]]; then
        [[ -L "$target" ]] && fail "$label must not be a symlink: $target"
        candidate="$(realpath -e "$target")" || fail "$label must resolve: $target"
    else
        candidate="${resolved_parent}/${base}"
    fi
    resolved_root="$(realpath -e "$root")" || fail "$label root must resolve: $root"
    case "$candidate" in
        "$resolved_root"/*) ;;
        *) fail "$label is not a strict descendant of its root: $target" ;;
    esac
    case "$candidate" in
        "$SCRIPT_DIR"|"$SCRIPT_DIR"/*) fail "$label is inside the repository: $target" ;;
    esac
    printf '%s\n' "$candidate"
}

WORK_DIR="${WORK_ROOT}/${SCENE}"
TRAJECTORY_DIR="${WORK_DIR}/trajectories"
RENDERED_DIR="${WORK_DIR}/rendered"
SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"

if [[ -e "$SCENE_OUTPUT" && $FORCE -ne 1 && $PLAN_ONLY -ne 1 ]]; then
    echo "ERROR: Output already exists: $SCENE_OUTPUT"
    echo "Use --force to replace this generated scene."
    exit 1
fi

# Shell-owned disposable work-directory reset, guarded like every deletion.
WORK_TARGET="$(guard_destructive_target "$WORK_DIR" "$WORK_ROOT" "WORK_DIR")"
echo "Resetting work directory: $WORK_TARGET"
rm -rf "$WORK_TARGET"
mkdir -p "$WORK_DIR"

echo "[1/4] Generating safe PointGoal trajectories for ${SCENE}"
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
"$ISAAC_PYTHON" -m sage3d.cli.generate \
    --scene "$SCENE" \
    --sage-root "$SAGE_ROOT" \
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

echo "[2/4] Allocating render staging and rendering RGB + depth"
RENDER_STAGE="$(PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PACKAGE_PYTHON" -m sage3d.cli.create_staging \
    --final-target "$RENDERED_DIR")"
[[ -n "$RENDER_STAGE" && -d "$RENDER_STAGE" ]] \
    || fail "create_staging returned an invalid staging path: $RENDER_STAGE"

for MODE in rgb depth; do
    PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$ISAAC_PYTHON" -m sage3d.cli.render \
        --scene "$SCENE" \
        --sage-root "$SAGE_ROOT" \
        --trajectory-dir "$TRAJECTORY_DIR" \
        --staging-root "$RENDER_STAGE" \
        --mode "$MODE" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --horizontal-fov-deg "$HORIZONTAL_FOV_DEG" \
        --fisheye-coefficients "${FISHEYE_COEFFICIENTS[@]}"
done

echo "Finalizing render"
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
"$PACKAGE_PYTHON" -m sage3d.cli.finalize_render \
    --scene "$SCENE" \
    --trajectory-dir "$TRAJECTORY_DIR" \
    --staging-root "$RENDER_STAGE" \
    --output-dir "$RENDERED_DIR"

echo "[3/4] Packaging LeRobot v2.1 PointGoal dataset"
# The production package CLI is non-destructive: it builds into a sibling
# staging directory and atomically renames onto the absent final target. Only
# the shell owns destructive replacement (--force removes the target before
# invoking the producer). Do NOT pre-create the final output dir.
if [[ $FORCE -eq 1 && -e "$SCENE_OUTPUT" ]]; then
    FORCE_TARGET="$(guard_destructive_target "$SCENE_OUTPUT" "$OUTPUT_ROOT" "SCENE_OUTPUT")"
    echo "Removing existing output: $FORCE_TARGET"
    rm -rf "$FORCE_TARGET"
fi
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
"$PACKAGE_PYTHON" -m sage3d.cli.package \
    --scene "$SCENE" \
    --trajectory-dir "$TRAJECTORY_DIR" \
    --rendered-dir "$RENDERED_DIR" \
    --output-dir "$SCENE_OUTPUT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --horizontal-fov-deg "$HORIZONTAL_FOV_DEG" \
    --fisheye-coefficients "${FISHEYE_COEFFICIENTS[@]}" \
    --camera-height "$CAMERA_HEIGHT"

echo "[4/4] Validating output inventory"
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
"$PACKAGE_PYTHON" "${SCRIPT_DIR}/scripts/check_package.py" validate \
    --dataset-dir "$SCENE_OUTPUT" \
    --trajectory-dir "$TRAJECTORY_DIR" \
    --rendered-dir "$RENDERED_DIR"

echo "DONE: ${SCENE_OUTPUT}"
