"""
Pre-extract trajectory poses from parquet files into simple .npy format.
This avoids needing pyarrow inside Blender's Python environment.

Output: for each episode, saves a .npy file containing:
  - extrinsic (4x4, constant per episode)
  - actions (N x 4x4, camera trajectory poses)
"""

import argparse
import json
import os
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", required=True, help="Extracted trajectory dir")
    parser.add_argument("--output_dir", required=True, help="Where to save .npy files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parquet_dir = os.path.join(args.traj_dir, "data", "chunk-000")
    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])

    print(f"[prepare] Found {len(parquet_files)} parquet files")

    for ep_idx, pf in enumerate(parquet_files):
        path = os.path.join(parquet_dir, pf)
        table = pq.read_table(path)

        extrinsics = np.array(table.column("observation.camera_extrinsic").to_pylist())
        actions = np.array(table.column("action").to_pylist())

        extrinsic = extrinsics[0].reshape(4, 4).astype(np.float64)
        actions = actions.reshape(-1, 4, 4).astype(np.float64)

        out_path = os.path.join(args.output_dir, f"episode_{ep_idx:06d}.npz")
        np.savez(out_path, extrinsic=extrinsic, actions=actions)
        print(f"[prepare] Episode {ep_idx}: {actions.shape[0]} frames -> {out_path}")

    print(f"[prepare] Done! {len(parquet_files)} episodes saved to {args.output_dir}")

    # Also write a summary file with frame counts
    summary = []
    npz_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith(".npz")])
    for ep_idx, nf in enumerate(npz_files):
        data = np.load(os.path.join(args.output_dir, nf))
        summary.append({"episode": ep_idx, "file": nf, "n_frames": data["actions"].shape[0]})
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
