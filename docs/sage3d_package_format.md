# SAGE3D Package Format Note

This note documents the **project-specific LeRobot-style** layout produced by
the SAGE3D packaging pipeline (`sage3d.lerobot_dataset.package`,
`python -m sage3d.cli.package`, and the legacy
`package_lerobot_sage3d.py` shim). It exists to make the format contract
explicit and to record that **standard LeRobot v2.1 loading is not claimed**.

## Status

This refactor is **behavior-preserving**, not a LeRobot format migration. The
current layout is a project-specific LeRobot-v2.1-style metadata layout:

- Images live under `videos/`, while `info["video_path"]` advertises MP4 paths.
- The image streams are absent from `features`.
- Standard LeRobot v2.1 loading (which expects a different `videos` layout and
  feature-keyed frame streams) is **not** supported by this format.

Converting to standard LeRobot is a separate post-refactor migration with its
own output-version boundary and data regeneration plan.

## Inventory

A packaged SAGE3D scene directory contains:

```
<output-dir>/
├── data/
│   └── chunk-000/
│       └── episode_<index:06d>.parquet     # one per episode
├── videos/
│   └── chunk-000/
│       ├── observation.images.rgb/
│       │   └── episode_<index:06d>_<frame:03d>.jpg
│       └── observation.images.depth/
│           └── episode_<index:06d>_<frame:03d>.png
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── episodes_stats.jsonl
    ├── pointcloud.ply                      # copied from trajectory dir
    ├── trajectory_manifest.json            # copied from trajectory dir
    ├── render_summary.json                 # canonical depth summary
    ├── rgb_render_summary.json
    └── depth_render_summary.json
```

## Authority

- **Calibration authority:** the canonical depth summary
  (`render_summary.json`) owns resolution, `horizontal_fov_deg`, and
  `fisheye_coefficients`. `package` constructs `CameraCalibration` from those
  fields and derives parquet intrinsic/extrinsic plus `info.json` camera
  metadata from it.
- **Camera height authority:** the manifest `camera_height_m` field.
- **Depth-encoding authority:** the canonical depth summary owns
  `depth_type`, `min_depth_m`, `max_depth_m`, and `depth_scale`. The
  `info.json` `depth_format` string is `uint16_meters_x_<scale>`; the default
  scale `10000.0` preserves the exact legacy string
  `uint16_meters_x_10000`.
- **Legacy CLI camera fields** (`--width`, `--height`,
  `--horizontal-fov-deg`, `--fisheye-coefficients`, `--camera-height`) are
  optional expected-value assertions checked by the staged validator; they do
  not override the authorities above.

## Non-Destructive Publication

Producers are non-destructive. The final output directory must be **absent**;
the package is built into an internally allocated sibling staging directory,
validated with `validate_packaged_dataset`, and atomically renamed onto the
target. Existing targets (including files, directories, symlinks, and dangling
symlinks) are refused. Only the shell owns destructive replacement
(`--force`), and it clears the target **before** invoking the producer.

## Validation

Every package run is validated before publication by
`validate_packaged_dataset` (inventory, Arrow/JSON content, copied-input
checksums, calibration/extrinsics, depth metadata) and can be independently
confirmed afterwards with `scripts/check_package.py`:

```bash
python scripts/check_package.py validate \
    --dataset-dir <output-dir> \
    --trajectory-dir <trajectory-dir> \
    --rendered-dir <rendered-dir>
```
