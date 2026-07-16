"""
Package rendered fisheye images + original trajectory data into LeRobot v2.1 format.

Reads:
  - Original extracted trajectory dir (parquet + meta)
  - Rendered fisheye images (RGB jpg + depth png)

Writes LeRobot v2.1 layout:
  output_dir/scene_name/
    data/chunk-000/episode_XXXXXX.parquet  (updated camera_intrinsic)
    meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, pointcloud.ply}
    videos/chunk-000/observation.images.rgb/episode_XXXXXX_YYY.jpg
    videos/chunk-000/observation.images.depth/episode_XXXXXX_YYY.png
"""

import argparse
import json
import os
import shutil
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def compute_fisheye_intrinsic(width, height, fov_deg):
    """
    Approximate pinhole-equivalent intrinsic for equidistant fisheye.
    For small angles, equidistant r=f*theta ~ pinhole r=f*tan(theta) ~ f*theta.
    Effective focal length: f = (width/2) / (fov_rad/2) = width / fov_rad
    """
    fov_rad = math.radians(fov_deg)
    f = width / fov_rad  # effective focal length in pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K


def update_parquet_files(traj_dir, output_dir, fisheye_K):
    """Copy parquet files, updating camera_intrinsic to fisheye params."""
    src_parquet_dir = os.path.join(traj_dir, "data", "chunk-000")
    dst_parquet_dir = os.path.join(output_dir, "data", "chunk-000")
    os.makedirs(dst_parquet_dir, exist_ok=True)

    K_flat = fisheye_K.flatten().tolist()
    parquet_files = sorted([f for f in os.listdir(src_parquet_dir) if f.endswith(".parquet")])

    for pf in parquet_files:
        src_path = os.path.join(src_parquet_dir, pf)
        dst_path = os.path.join(dst_parquet_dir, pf)

        table = pq.read_table(src_path)
        n_rows = table.num_rows

        # Replace camera_intrinsic column with fisheye K (same for all rows)
        new_intrinsics = pa.array([K_flat] * n_rows, type=table.schema.field("observation.camera_intrinsic").type)
        table = table.set_column(
            table.schema.get_field_index("observation.camera_intrinsic"),
            "observation.camera_intrinsic",
            new_intrinsics
        )

        pq.write_table(table, dst_path)

    print(f"[package] Updated {len(parquet_files)} parquet files with fisheye intrinsic")
    return len(parquet_files)


def copy_meta_files(traj_dir, output_dir, n_episodes, width, height, fov_deg):
    """Copy meta files, updating info.json with fisheye camera info."""
    src_meta = os.path.join(traj_dir, "meta")
    dst_meta = os.path.join(output_dir, "meta")
    os.makedirs(dst_meta, exist_ok=True)

    # Copy pointcloud.ply, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
    for fname in ["pointcloud.ply", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"]:
        src = os.path.join(src_meta, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_meta, fname))
            print(f"[package] Copied {fname}")

    # Update info.json
    info_path = os.path.join(src_meta, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    # Update camera info
    info["robot_type"] = "fisheye_navdp"
    info["camera_model"] = "fisheye_equidistant"
    info["camera_fov_deg"] = fov_deg
    info["image_width"] = width
    info["image_height"] = height

    with open(os.path.join(dst_meta, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[package] Updated info.json with fisheye camera info")


def link_rendered_images(rendered_dir, output_dir):
    """Move/symlink rendered images into the LeRobot videos/ directory."""
    rgb_src = os.path.join(rendered_dir, "observation.images.rgb")
    depth_src = os.path.join(rendered_dir, "observation.images.depth")

    rgb_dst = os.path.join(output_dir, "videos", "chunk-000", "observation.images.rgb")
    depth_dst = os.path.join(output_dir, "videos", "chunk-000", "observation.images.depth")
    os.makedirs(rgb_dst, exist_ok=True)
    os.makedirs(depth_dst, exist_ok=True)

    n_rgb = 0
    if os.path.exists(rgb_src):
        for f in os.listdir(rgb_src):
            shutil.copy2(os.path.join(rgb_src, f), os.path.join(rgb_dst, f))
            n_rgb += 1

    n_depth = 0
    if os.path.exists(depth_src):
        for f in os.listdir(depth_src):
            shutil.copy2(os.path.join(depth_src, f), os.path.join(depth_dst, f))
            n_depth += 1

    print(f"[package] Linked {n_rgb} RGB + {n_depth} depth images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--traj_dir", required=True, help="Extracted original trajectory dir")
    parser.add_argument("--rendered_dir", required=True, help="Dir with rendered fisheye images")
    parser.add_argument("--output_dir", required=True, help="Final output dir for this scene")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov_deg", type=float, default=195.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fisheye_K = compute_fisheye_intrinsic(args.width, args.height, args.fov_deg)
    print(f"[package] Fisheye intrinsic matrix:\n{fisheye_K}")

    n_eps = update_parquet_files(args.traj_dir, args.output_dir, fisheye_K)
    copy_meta_files(args.traj_dir, args.output_dir, n_eps, args.width, args.height, args.fov_deg)
    link_rendered_images(args.rendered_dir, args.output_dir)

    print(f"[package] Done! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
