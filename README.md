# Fisheye VLN Data Preparation for NavDP

Renders 195-degree equidistant fisheye RGB and depth images from InternData-N1
trajectory poses and packages complete runs in LeRobot v2.1 format for NavDP.

Supported scene sources:

| Dataset | Render asset | Pipeline | Output root |
|---|---|---|---|
| Replica | Vertex-colored `mesh.ply` | `run_pipeline.sh` | `/ssd5/datasets/vln-fisheye/replica` |
| HSSD | Composite stage/object GLBs | `run_pipeline_hssd.sh` | `/ssd5/datasets/vln-fisheye/hssd` |
| Gibson V2 | UV-textured `<scene>_mesh_texture.obj` | `run_pipeline_gibson.sh` | `/ssd5/datasets/vln-fisheye/gibson` |
| HM3D 0.2 | Self-contained textured scene GLB | `run_pipeline_hm3d.sh` | `/ssd5/datasets/vln-fisheye/hm3d` |
| MP3D / Scene-N1 | Textured Matterport OBJ/MTL/JPEG tiles | `run_pipeline_mp3d.sh` | `/ssd5/datasets/vln-fisheye/mp3d` |
| 3D-FRONT | Scene JSON + 3D-FUTURE furniture + architecture textures | `run_pipeline_3dfront.sh` | `/ssd5/datasets/vln-fisheye/3dfront` |

## Quick Start

```bash
# Replica
bash run_pipeline.sh apartment_1

# HSSD
bash run_pipeline_hssd.sh 102344049

# Gibson: sparse first-episode validation (not packaged as LeRobot)
bash run_pipeline_gibson.sh Sodaville --smoke-test --samples 4

# Gibson: render every frame and package the complete scene
bash run_pipeline_gibson.sh Sodaville

# HM3D: sparse validation, then a complete packaged scene
bash run_pipeline_hm3d.sh 00275-4dbCzNN5L5t --smoke-test --samples 4
bash run_pipeline_hm3d.sh 00275-4dbCzNN5L5t

# MP3D: sparse validation, then a complete packaged scene
bash run_pipeline_mp3d.sh 29hnd4uzFmX --smoke-test --samples 4
bash run_pipeline_mp3d.sh 29hnd4uzFmX

# 3D-FRONT: sparse validation, then a complete packaged scene
bash run_pipeline_3dfront.sh fbe0dae7-7c8a-4a3f-aac9-082d5c509469 --smoke-test --samples 4
bash run_pipeline_3dfront.sh fbe0dae7-7c8a-4a3f-aac9-082d5c509469
```

Sparse smoke-test images are stored under
`/ssd5/datasets/vln-fisheye/<dataset>-smoke/<scene>/`. Smoke tests intentionally
do not create LeRobot datasets because their frame stride is 20.

## Gibson V2 Setup

The complete textured archive is expected at:

```text
/ssd5/datasets/gibson/gibson-v2-all/gibson_v2_all.tar.gz
```

Verify and extract it once:

```bash
gzip -t /ssd5/datasets/gibson/gibson-v2-all/gibson_v2_all.tar.gz

tar -I 'pigz -p 16' -xf \
  /ssd5/datasets/gibson/gibson-v2-all/gibson_v2_all.tar.gz \
  -C /ssd5/datasets/gibson/gibson-v2-all
```

The extracted `gibson_v2/` directory contains 572 Gibson scenes plus the separate
Stanford 2D3DS `area1` asset. Each Gibson scene has:

```text
gibson_v2/<scene>/
├── <scene>_mesh_texture.obj       # high-detail UV render mesh
├── <scene>_mesh_texture.obj.mtl   # diffuse material
├── <scene>_mesh_texture.png       # texture atlas
├── <scene>_mesh_texture.ply
├── floor_*.png / floor_trav_*.png
└── metadata.json
```

The rendering pipeline uses `<scene>_mesh_texture.obj`. Do not substitute a
simplified collision mesh from a legacy Gibson distribution: it has no RGB
texture and is not included for ordinary scenes in the extracted V2 archive.

### Fetching Gibson Trajectories

The local InternData-N1 clone initially contains 134-byte Git-LFS pointers. Fetch
only the scenes that will be rendered:

```bash
cd /ssd5/datasets/InternData-N1

git lfs pull \
  --include="vln_n1/traj_data/gibson_zed/Sodaville.tar.gz" \
  --exclude=""
```

`run_pipeline_gibson.sh` detects unresolved pointers and prints the corresponding
selective `git lfs pull` command instead of passing them to `tar`.

To process all already-downloaded Gibson Zed archives:

```bash
bash process_all_gibson.sh
```

Logs are written to `/tmp/opencode/gibson_batch_logs/`. Unresolved LFS pointers
are reported as skipped/failed scenes rather than downloaded automatically.

## HM3D 0.2 Setup

The extracted textured assets are expected at:

```text
/ssd5/datasets/hm3d/versioned_data/hm3d-0.2/hm3d/
├── train/<numeric-prefix>-<scene-id>/<scene-id>.glb
└── val/<numeric-prefix>-<scene-id>/<scene-id>.glb
```

The local download contains 800 training and 100 validation GLBs. All 633
scene names in `hm3d_zed` match one of those assets. The trajectory clone may
still contain Git-LFS pointer files; fetch only the scenes to process:

```bash
cd /ssd5/datasets/InternData-N1

git lfs pull \
  --include="vln_n1/traj_data/hm3d_zed/00275-4dbCzNN5L5t.tar.gz" \
  --exclude=""
```

`run_pipeline_hm3d.sh` resolves the GLB from either split, verifies the
trajectory gzip, rejects unresolved LFS pointers, renders, and packages complete
runs. Its `--smoke-test` mode renders every 20th frame from episode 0 into
`/ssd5/datasets/vln-fisheye/hm3d-smoke/<scene>/` and skips packaging because the
frames are sparse.

To process every HM3D Zed trajectory that has already been downloaded:

```bash
bash process_all_hm3d.sh
```

Logs are written to `/tmp/opencode/hm3d_batch_logs/`. Pointer files are skipped
before invoking the per-scene pipeline, so this command does not fetch data.

## MP3D / Scene-N1 Setup

The extracted Scene-N1 assets are expected at:

```text
/ssd5/datasets/Scene-N1/mp3d_n1/<scene>/
├── matterport_mesh/<mesh-id>/
│   ├── <mesh-id>.obj
│   ├── <mesh-id>.mtl
│   ├── <mesh-id>_NNN.jpg       # texture tiles referenced by the MTL
│   └── textures/               # duplicate texture-tile directory
└── house_segmentations/        # semantic assets; not needed for RGB/depth
```

The local asset directory contains 90 scenes, each with exactly one render OBJ
and MTL. All 66 `matterport3d_zed` trajectory scene IDs match those assets.
Trajectory archives may still be Git-LFS pointers, so fetch only the desired
scenes:

```bash
cd /ssd5/datasets/InternData-N1

git lfs pull \
  --include="vln_n1/traj_data/matterport3d_zed/29hnd4uzFmX.tar.gz" \
  --exclude=""
```

`run_pipeline_mp3d.sh` resolves the OBJ/MTL automatically, verifies adjacent
JPEG textures and trajectory gzip integrity, rejects unresolved LFS pointers,
and renders/packages complete runs. `--smoke-test` renders every 20th frame from
episode 0 under `/ssd5/datasets/vln-fisheye/mp3d-smoke/<scene>/` without making
an incomplete LeRobot package.

To process every MP3D Zed trajectory that has already been downloaded:

```bash
bash process_all_mp3d.sh
```

Logs are written to `/tmp/opencode/mp3d_batch_logs/`. Pointer files are skipped
without downloading them.

## 3D-FRONT Setup

Keep the six official download archives at:

```text
/ssd5/datasets/3dfront/
├── 3D-FRONT.zip
├── 3D-FRONT-texture.zip
├── 3D-FUTURE-model-part1.zip
├── 3D-FUTURE-model-part2.zip
├── 3D-FUTURE-model-part3.zip
└── 3D-FUTURE-model-part4.zip
```

No full extraction is required. `prepare_3dfront_assets.py` reads the archive
indexes and extracts only one scene's JSON, referenced 3D-FUTURE furniture, and
architecture textures into:

```text
/ssd5/datasets/3dfront/prepared/<scene>/
├── <scene>.json                  # normalized for BlenderProc 2.8
├── manifest.json
├── 3D-FUTURE-model/<model-id>/
└── 3D-FRONT-texture/<texture-id>/
```

The preparation is idempotent. The normalized JSON also handles newer
3D-FRONT releases that put architecture texture UUIDs in material `jid` fields
while leaving BlenderProc's legacy `texture` URL empty. Materials marked
`useColor=true` remain solid-color.

All 661 local `3dfront_zed` trajectory scene IDs have corresponding JSON files.
Trajectory archives initially stored as Git-LFS pointers can be fetched
selectively:

```bash
cd /ssd5/datasets/InternData-N1

git lfs pull \
  --include="vln_n1/traj_data/3dfront_zed/fbe0dae7-7c8a-4a3f-aac9-082d5c509469.tar.gz" \
  --exclude=""
```

`run_pipeline_3dfront.sh` verifies the six asset archives are present, rejects
unresolved trajectory pointers, validates the gzip, prepares only the required
assets, checks all camera positions against assembled scene bounds, renders in
episode-sized batches, and packages complete runs. Episode batching prevents
large 3D-FRONT scenes from retaining thousands of float RGB/depth frames and
temporary EXRs in memory at once.

Some official scene JSONs reference furniture models that are not distributed
in the public 3D-FUTURE archives. The asset manifest records those UUIDs and the
pipeline emits a warning; BlenderProc still renders the available architecture
and furniture, consistent with its official 3D-FRONT loader behavior.

To process every already-downloaded 3D-FRONT Zed trajectory:

```bash
bash process_all_3dfront.sh
```

Logs are written to `/tmp/opencode/3dfront_batch_logs/`. Git-LFS pointers are
skipped without downloading them.

## Pipeline Overview

```text
InternData trajectory tar.gz
          │
          ├── 1. Extract original LeRobot scene
          ├── 2. Convert parquet action poses to NPZ
Scene ────┤
asset     ├── 3. Render fisheye RGB/depth with BlenderProc
          └── 4. Package complete runs as LeRobot v2.1
```

Shared components:

- `prepare_trajectories.py` reads parquet outside Blender and writes episode NPZ files.
- `package_lerobot.py` copies metadata, replaces the camera intrinsic, and assembles output.
- `render_fisheye.py` handles Replica vertex-color PLY scenes.
- `render_fisheye_hssd.py` assembles HSSD GLB stages and object instances.
- `render_fisheye_gibson.py` imports textured Gibson V2 OBJ/MTL/PNG scenes.
- `render_fisheye_hm3d.py` imports self-contained textured HM3D GLB scans.
- `render_fisheye_mp3d.py` imports tiled-texture Scene-N1 Matterport OBJ scans.
- `prepare_3dfront_assets.py` selectively extracts and normalizes one 3D-FRONT scene.
- `render_fisheye_3dfront.py` assembles 3D-FRONT architecture and 3D-FUTURE furniture.

## Camera Configuration

| Parameter | Value |
|---|---|
| Model | Equidistant fisheye |
| FOV | 195 degrees |
| Resolution | 640×640 |
| Effective focal length | ~188 px (`width / FOV_rad`) |
| Principal point | (320, 320) |
| Depth format | uint16, meters × 10000 |
| Depth clip | 0–6.0 m |
| RGB format | JPEG, quality 95 |

## Coordinate and Material Conventions

### Camera Poses

The `action` column is used directly as the Blender camera-to-world matrix. It
has been validated against scene bounds and original rendered frames for all
supported datasets.

### Gibson V2 Axes

The textured Gibson OBJ and InternData actions use the same Z-up world frame.
Blender 4.2's OBJ importer is called with `forward_axis=Y, up_axis=Z`; using
`forward_axis=-Y` incorrectly rotates the scene 180 degrees around Z. The Gibson
renderer checks every selected camera position against the imported world-space
mesh bounds and aborts before rendering if they disagree.

### Gibson V2 Materials

The OBJ's diffuse texture atlas contains the captured scene appearance. The
renderer preserves the texture but feeds it through an emission shader so that
Cycles does not apply a second artificial lighting pass. It also overrides the
legacy MTL `Tr 1` transparency field. Rendering fails explicitly if the MTL or
texture image was not loaded.

### HM3D Axes and Materials

Habitat-ready HM3D GLBs store coordinates in the same Z-up world used by the
InternData actions, despite glTF's nominal Y-up convention. Blender's glTF
importer automatically maps `(x,y,z)` to `(x,-z,y)`, so the HM3D renderer
immediately applies the inverse `(x,y,z)` to `(x,z,-y)` to imported root
objects. It then checks every selected camera against the corrected world-space
scene bounds and aborts on a mismatch.

HM3D diffuse textures contain the captured scan appearance. The renderer keeps
the imported base-color node graph but routes it through emission, preventing an
artificial second lighting pass. It fails if no textured base-color material is
found.

### MP3D Axes and Materials

Scene-N1 MP3D OBJ vertices and InternData actions use the same Z-up world frame.
Blender 4.2 imports the OBJ with `forward_axis=Y, up_axis=Z`, preserving the raw
coordinates. Every selected camera is checked against the imported world-space
bounds before rendering.

The MP3D MTL files contain both `map_Ka` and `map_Kd` entries for each JPEG tile.
Blender warns that ambient `map_Ka` is unsupported but correctly loads the
diffuse `map_Kd` image; this warning is harmless. As with HM3D and Gibson, the
captured diffuse texture graph is routed through emission so no artificial
lighting pass changes the scan appearance.

### 3D-FRONT Axes and Materials

Official 3D-FRONT JSON geometry and furniture placements are Y-up. BlenderProc's
`load_front3d` loader converts them to Blender Z-up by swapping Y and Z.
InternData `3dfront_zed` actions use that resulting Z-up world frame directly.
The renderer checks all selected poses against assembled world-space bounds
before rendering.

3D-FRONT is synthetic rather than a captured scan, so it retains Principled PBR
materials and uses the ceiling/lamp emission created by the official loader,
plus a low-strength world background. Legacy OBJ transmission and unintended
alpha are disabled. The local official source textures need not match the
original InternData RGB exactly because InternData generation applied texture,
lighting, and view randomization; geometry and pose alignment are the invariant
validation targets.

### Replica Materials

Replica's `mesh.ply` stores pre-baked vertex colors. It is rendered with a
vertex-color emission material and the Standard view transform.

### HSSD Materials

HSSD assets use PBR materials and separate object instances. Its pipeline
assembles the Habitat scene description, decompresses BasisU textures when
needed, and provides scene lighting.

## Output Structure

Complete runs produce:

```text
/ssd5/datasets/vln-fisheye/<dataset>/<scene>/
├── data/chunk-000/
│   └── episode_XXXXXX.parquet
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   └── pointcloud.ply
└── videos/chunk-000/
    ├── observation.images.rgb/
    │   └── episode_XXXXXX_YYY.jpg
    └── observation.images.depth/
        └── episode_XXXXXX_YYY.png
```

## Dependencies

- Python environment: `/ssd4/envs/vln_data_prep_py311` for HM3D, MP3D, and 3D-FRONT;
  `/ssd4/envs/vln_py311` for the earlier pipelines
- BlenderProc 2.8.0 / Blender 4.2.1
- Python packages: `pyarrow`, `Pillow`, `numpy`, `pandas`, `jsonlines`
- NVIDIA GPU with Cycles/OptiX support
- `pigz` is optional but substantially speeds up Gibson V2 extraction

## Validation Results

### 3D-FRONT

- Archive inventory: one scene ZIP, one architecture-texture ZIP, and all four
  3D-FUTURE model ZIP parts are present. All 661 `3dfront_zed` scene IDs match a
  JSON member in `3D-FRONT.zip`.
- Selective asset preparation passed for
  `fbe0dae7-7c8a-4a3f-aac9-082d5c509469` (5/5 used furniture models and two
  architecture textures) and `9ed1d34e-0432-4430-a6a5-dd81cd233873` (2/2
  furniture models and three architecture textures).
- Four-sample smoke tests produced eight and nine sparse frame pairs,
  respectively. The loader-converted Z-up geometry, camera headings, furniture,
  and room boundaries visually align with the corresponding original Zed
  frames; source texture appearance differs where InternData applied domain
  randomization.
- The complete four-sample validation scene contains 6,330 RGB/depth pairs
  across 43 episodes. Rendering in episode-sized batches took approximately
  2,129 seconds (35.5 minutes); the packaged output occupies about 1.9 GB.
- All 6,330 source camera positions passed the assembled scene-bounds check.
  Source/output parquets preserve every action, extrinsic, index, and other
  non-intrinsic column exactly; only the intrinsic was replaced.
- Exact image filename sets and counts passed. Every RGB image is 640×640 RGB,
  every depth image is 640×640 uint16, the observed stored depth range is
  2,564–60,000, metadata contains all 43 episodes, and unresolved Git-LFS
  pointers fail with the selective-fetch command.
- Complete output:
  `/ssd5/datasets/vln-fisheye/3dfront/fbe0dae7-7c8a-4a3f-aac9-082d5c509469`.

### MP3D / Scene-N1

- Asset inventory: 90/90 scenes contain exactly one textured OBJ/MTL pair.
- Trajectory coverage: 66/66 `matterport3d_zed` scene IDs match an asset.
- Selective trajectory gzip validation passed for `29hnd4uzFmX` and
  `RPmz2sHmrrY`.
- Smoke tests rendered four sparse frames per scene and visually aligned with
  the corresponding original Zed views.
- Full pose checks accepted 1,814/1,814 and 1,649/1,649 cameras. All 43 and 28
  diffuse materials, respectively, loaded textured.
- Complete 4-sample renders contain 1,814 RGB/depth pairs across 15 episodes and
  1,649 pairs across 21 episodes. Rendering took approximately 611 and 553
  seconds; packaged outputs occupy about 700 MB and 649 MB.
- Source/output parquets preserve every action and extrinsic value; only camera
  intrinsics changed. Image names/counts, 640×640 RGB, uint16 depth with 6 m
  clipping, metadata totals, and LFS-pointer rejection all passed validation.
- Complete outputs: `/ssd5/datasets/vln-fisheye/mp3d/29hnd4uzFmX` and
  `/ssd5/datasets/vln-fisheye/mp3d/RPmz2sHmrrY`.

### HM3D 0.2

- Asset inventory: 800 train + 100 val textured GLBs.
- Trajectory coverage: 633/633 `hm3d_zed` scene IDs match an asset.
- Selective trajectory gzip validation passed for two sample scenes.
- Smoke tests passed on `00275-4dbCzNN5L5t` (13 sparse frames) and
  `00190-NkvRYHk72vA` (10 sparse frames), including visual alignment with the
  corresponding original Zed views.
- Corrected bounds accepted 1,546/1,546 and 1,712/1,712 camera poses,
  respectively; the uncorrected Blender glTF conversion is rejected.
- Complete 4-sample validation renders contain 1,546 RGB/depth pairs across 8
  episodes and 1,712 pairs across 11 episodes. Rendering took approximately
  525 and 581 seconds, and the packaged outputs occupy about 506 MB and 466 MB.
- Source and output parquet files have byte-equivalent action/extrinsic values;
  only camera intrinsics were replaced. All image names/counts, 640×640 RGB and
  uint16 depth formats, 6 m clipping, metadata episode counts, and the
  unresolved-LFS-pointer failure path passed validation.
- Complete outputs: `/ssd5/datasets/vln-fisheye/hm3d/00275-4dbCzNN5L5t` and
  `/ssd5/datasets/vln-fisheye/hm3d/00190-NkvRYHk72vA`.

### Gibson V2 / Sodaville

- Archive integrity: passed full gzip CRC validation.
- Asset coverage: 572/572 textured OBJ/MTL/PNG scene triples.
- InternData coverage: 506/506 Zed scene IDs match a V2 scene asset.
- Selected trajectory: 9 episodes, 1,998 frames.
- Smoke render: 9 frames (`episode_000000`, stride 20), 640×640, 4 samples.
- RGB texture, fisheye circle, depth output, pose bounds, and viewpoint alignment
  against the original Zed images were visually verified.
- Complete render: 1,998/1,998 RGB and depth pairs, 9/9 parquet files, 4 samples.
- Render time: 692 seconds; packaged output size: 608 MB.
- Complete output: `/ssd5/datasets/vln-fisheye/gibson/Sodaville`.

### Replica / apartment_1

- 18 episodes, 1,565 frames.
- Approximately 12 minutes at 640×640 and 16 Cycles samples.
- Approximately 528 MB output.
