# SAGE3D Refactor Plan (revision 5)

Consolidate `generate_sage3d_trajectories.py`, `render_fisheye_sage3d.py`, and
`package_lerobot_sage3d.py` into a single `sage3d/` package, eliminating
cross-script duplication and making each stage independently testable.

> **Status:** plan only — no code written yet. Five full review passes
> incorporated. **Revision 5 closes the remaining handoff ambiguities:** fixes
> the Phase 3 shell/absent-target conflict, fully specifies depth encoding and
> streaming raw-depth summaries, pins float comparison classes, chooses exact
> render/finalizer CLIs and `PYTHONPATH` discovery, defines legacy-render shim
> behavior and single-publisher semantics, corrects dependency annotations,
> formalizes Phase 0b provenance/re-estimation, and identifies
> `prepare_trajectories.py` as a partial schema smoke test only.

---

## How the three scripts connect today

```
generate_sage3d_trajectories.py
   └─► trajectories/episode_*.npz  {points, actions, camera_positions, yaw,
                                    point_goal, start_position, goal_position}
   └─► trajectories/trajectory_manifest.json
   └─► trajectories/pointcloud.ply
   └─► trajectories/navigation_map.png, trajectories_overlay.png
        │
        ▼
render_fisheye_sage3d.py   (invoked twice: --mode rgb, then --mode depth)
   └─► reads npz[camera_positions, yaw]
   └─► rendered/observation.images.rgb/*.jpg          (lossy JPEG, quality=95)
   └─► rendered/observation.images.depth/*.png        (uint16, meters × depth_scale)
   └─► rendered/rgb_render_summary.json
   └─► rendered/depth_render_summary.json  AND  rendered/render_summary.json
        (depth mode writes the SAME summary to both filenames)
        │
        ▼
package_lerobot_sage3d.py
   └─► reads npz[actions, point_goal], manifest, render_summary.json (depth),
        rgb_render_summary.json; depth_render_summary.json is existence-checked
        and copied but NOT loaded
   └─► cross-validates counts / resolution / fisheye / focal length
   └─► copies pointcloud.ply + all summaries into meta/
   └─► project-specific LeRobot-v2.1-style dataset
        (parquet + individual images + meta/*.json[l])
```

**Render-summary policy:** `render_summary.json` is **canonical** for depth (it
is the file `package` loads); `depth_render_summary.json` is an alias that must
be **semantically equal** while both exist, removed in a later compatibility
change.

---

## Ground rules

### Environment split (drives module boundaries)

| Script   | Runs under                | Heavy deps allowed                          |
| -------- | ------------------------- | ------------------------------------------- |
| generate | Isaac python (`pxr`)      | cv2, scipy, trimesh, pxr                    |
| render   | Isaac python (`isaacsim`) | isaacsim, pxr                               |
| package  | plain python (`pyarrow`)  | numpy, pyarrow, PIL **only**                |

**`config.py` must be entirely package-safe** (stdlib + `pathlib` only — no
numpy/cv2/scipy/trimesh/pxr/isaacsim/PIL/pyarrow). Python imports the *whole*
module, so "the `PackageConfig` part" is not a meaningful boundary. "Isaac side"
describes who *consumes* a config class, not what `config.py` imports —
**therefore every config class (`RenderConfig` included) is package-safe**;
`RenderConfig.mode` is a plain `str`/`Literal`, **not** `RenderMode` imported
from `render_runtime.py`.

Package-safe modules (importable by `package`): `config`, `frames`, `camera`,
`episode_arrays`, `naming`, `schemas`, `contract` (numpy + PIL), `io_ply`
(numpy), `pointcloud` (numpy), `artifacts` and `publication` (stdlib + pathlib),
`render_processing` (numpy), `lerobot_dataset`.

### Binding forbidden-import smoke test (incremental — corrected)

Run in a **fresh subprocess** (so preloaded pytest plugins can't populate
`sys.modules`) and assert `cv2`/`scipy`/`trimesh`/`pxr`/`isaacsim` are absent.
**Import only the modules that exist at the running phase** (rev 2 wrongly
listed `config`/`schemas`/`contract`/`lerobot_dataset` in the Phase 1 DoD before
they existed):
- **Phase 1:** `frames, camera, episode_arrays, naming, io_ply, pointcloud,
  publication, render_processing, sage3d.cli._args`
- **Phase 2a adds:** `config, schemas`
- **Phase 2b adds:** `contract`
- **Phase 2c adds:** `artifacts`
- **Phase 4 adds:** `sage3d.cli.finalize_render`
- **Phase 5 adds:** `lerobot_dataset`

### Determinism (precise)
- `generate`: seed-deterministic only under a pinned environment + fixed input
  asset bytes.
- `package`: semantically deterministic given fixed inputs (compare tables, not
  Parquet bytes).
- `render`: GPU-nondeterministic → structural + logic + tolerant-frame oracle.

### Parallelism / batching — explicit non-goal during refactor
No parallel rejection sampling, frames-in-one-app, or RGB+depth concurrency.
Preserve `collision_distances` `2048` batching (collision-query batch size, not
pathfinding). Cached proximity is a Phase 7 benchmark item.

### Scope boundary — explicit

This refactor covers only `generate_sage3d_trajectories.py`,
`render_fisheye_sage3d.py`, `package_lerobot_sage3d.py`, and their SAGE3D shell
orchestration. The generic `package_lerobot.py`, `prepare_trajectories.py`, and
all non-SAGE3D `render_fisheye_*.py` / `run_pipeline_*.sh` implementations are
not folded into `sage3d/` and remain behaviorally untouched. Phase 5 may invoke
`prepare_trajectories.py` as a **supplemental shared-Parquet schema smoke test**;
it is not the authoritative SAGE3D consumer because it reads only
`observation.camera_extrinsic` and `action`, not SAGE3D point-goal, render, or
metadata contracts.

---

## Oracles (precise; tolerances pinned from the Phase 0b baseline)

### Generation extraction oracle (behavior-preserving phases)
- Exact npz **key set**, **shape**, **dtype**; `np.array_equal` for arrays.
- Recursive manifest comparison: exact strings/ints/bools/list-lengths/dict-keys;
  **exact parsed float equality** (not "where possible"); **explicit dict
  key-order comparison**; **path-field normalization limited to exactly these
  fields**: `scene_dir`, `collision_usd` (no heuristic path-string munging).
- Exact episode inventory (no missing/extra).
- **`pointcloud.ply`: exact bytes** in every writer-touching phase + header/
  vertex-count/dtype/bounds + cross-check vs `manifest["pointcloud"]`.
- `navigation_map.png` / `trajectories_overlay.png`: decoded-pixel equality
  (or checksum) + dimensions.

### Package oracle
Semantic Arrow **schema + table** equality (column order, types, row count,
values); parsed JSON/JSONL equality incl record order; copied RGB/depth +
pointcloud + summaries equal inputs (checksums); exact output inventory.

**Post-package extrinsic/calibration checks live in `check_package.py`** (not in
the pre-package validator — corrected, see Cross-artifact validation): manifest
`camera_height_m` ↔ every packaged parquet `observation.camera_extrinsic`
(shape `[4,4]`, identity rotation, `[2,3] == camera_height_m`, equal across
frames); render calibration ↔ packaged `info.json` intrinsic/distortion and
parquet intrinsic/extrinsic/distortion.

### Render oracle (corrected, tolerances pinned)
Pre-encode + decoded split (decoded JPEG outside the mask is **not** exactly
black — lossy q95). Concrete rules, with numeric thresholds **established from
the Phase 0b baseline**, not chosen by the Phase 4 implementer:
1. **Pre-encode RGB unit test:** `rgb[~mask] == 0` exactly.
2. **Decoded-JPEG mask leakage:** mean intensity (per channel) outside a mask
   **dilated by N pixels** (N pinned from baseline) ≤ threshold T₁ (pinned).
3. **Golden frames:** masked RMSE/SSIM against selected **decoded baseline
   JPEGs** (not raw frames), ≤ threshold T₂ (pinned), on the pinned GPU setup.
4. **`encode_depth` exact synthetic matrix:** NaN, ±inf, below/exactly-at min,
   ordinary valid, exactly-at/above max, inside/outside mask, `np.rint` half-step
   behavior, integral/non-integral scale, overflow, shape/mask mismatch, exact
   dtype/shape, and input non-mutation (contract below).
5. **Depth PNG structural checker:** inventory, shape, `uint16` dtype, encoded
   bounds, outside-mask encoded value `round(max_depth_m * depth_scale)`; **do
   not** compare encoded min/max against the raw summary's
   `finite_depth_min/max`.
6. **Selected golden depth frames:** compare decoded PNGs against the Phase 0b
   baseline using valid/non-max mask agreement plus pinned median/p95/p99
   absolute encoded-depth error limits. Tolerances are characterized on the
   pinned GPU and must detect all-max or otherwise corrupted inside-mask output.
7. **Raw-depth accumulator:** exact pure unit tests on synthetic float arrays
   and streaming/error behavior (contract below).
8. **Mocked stage-construction test:** assert prim `/World/gauss` with exact
   reference string incl `[gauss.usda]`; depth prim `/World/scene_collision`
   with exact collision-payload path; default prim `/World`.
9. **Complete render call-trace test:** exact stage/camera sequence and pose
   order: build stage → `World.reset()` → global startup steps → construct and
   initialize camera → set clipping/calibration → calibration readback → attach
   depth annotator when applicable → second global startup steps → per-episode
   poses. The first frame of **every episode** uses startup steps; later frames
   use settle steps.
10. **Depth-overflow precondition** (`max_depth_m * depth_scale ≤ 65535`): in
   force from Phase 1 onward — `render_processing.encode_depth` **raises on
   overflow** (fail-fast); `RenderConfig` (Phase 4) formalizes the validated
   constraint. This is an intentional new guard for invalid legacy inputs, not
   a behavior-preservation expectation from the unmodified Phase 0b baseline.
   Defaults `6.0 × 10000 = 60000` remain valid.
- `scripts/check_render.py` added in Phase 0b and **run in every render-touching
  phase (1, 2a, 2c, 4)**.

#### `encode_depth` contract (specified)

```python
encode_depth(
    depth: np.ndarray,
    circular_mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    depth_scale: float,
) -> np.ndarray  # shape == depth.shape, dtype == np.uint16
```

- Require numeric two-dimensional `depth` and `circular_mask.dtype == np.bool_`
  with the exact same shape; reject mismatch before producing output. Process a
  private `np.float32` copy of depth to preserve current operation precision.
- Reject non-finite/non-positive scale, invalid depth range, or
  `max_depth_m * depth_scale > 65535` before mutating/allocating the result.
- Valid raw pixels = finite and `depth >= min_depth_m` and inside the mask.
- NaN, ±inf, below-min, and outside-mask pixels become `max_depth_m`.
- Valid values above max are clipped to `max_depth_m` for encoding only.
- Encode with `np.rint(encoded_meters * depth_scale).astype(np.uint16)`; preserve
  NumPy half-to-even behavior exactly.
- Never mutate `depth` or `circular_mask`.

#### Streaming raw-depth summary contract (specified)

Use one accumulator per episode; do not retain or materialize raw frames:

```python
summary = RawDepthSummaryAccumulator(circular_mask, min_depth_m)
for depth in depth_frames:
    summary.add(depth)
result: RawDepthEpisodeSummary = summary.finish()
```

`add` requires a two-dimensional frame matching the non-empty mask and raises
when the frame contains no valid depth, preserving the current renderer.
`finish` raises when zero frames were added. Preserve the current four aggregate
formulas exactly:
- per-frame valid mask = `np.isfinite(depth) & (depth >= min_depth_m) & circular_mask`;
- per-frame finite fraction = `valid_inside.sum() / circular_mask.sum()`;
- episode `finite_depth_fraction_mean`/`_min` = mean/min of **per-frame**
  fractions (not one global fraction — matters when frame masks vary);
- episode `finite_depth_min_m` = min over per-frame raw minima;
  `finite_depth_max_m` = max over per-frame raw maxima.
- Raw valid depths **above `max_depth_m` are included** in
  `finite_depth_max_m` (clipping is for PNG encoding only).

---

## Cross-artifact validation (preserved + strengthened; split corrected)

`contract.py::validate_pipeline_contract(...)` runs **pre-package** (in `package`
before any parquet is written) and therefore **cannot** inspect packaged
extrinsics/calibration — those move to `check_package.py`. Explicit signature:
```
validate_pipeline_contract(
    expected_scene_id,
    manifest,
    rgb_summary,
    canonical_depth_summary,   # render_summary.json
    depth_alias_summary,       # depth_render_summary.json (verified equal)
    episodes_by_id,            # {episode_id: EpisodeArrays}
    trajectory_dir,
    rendered_dir,
    pointcloud_path,
)
```
Pre-package invariants (explicit exceptions, not `assert`): manifest episode
count ↔ npz inventory; manifest frame counts ↔ npz lengths; RGB ↔ depth frame
counts; **RGB ↔ canonical-depth calibration agreement**; scene IDs across
CLI/manifest/summaries; contiguous episode indexes ↔ filename stems
(`naming.parse_*`); image inventory + dims/dtype; no extra/stale frames;
**manifest `camera_height_m` ↔ NPZ `camera_positions[:, 2]`** (the packaged
extrinsic is checked post-package). Package invokes this only on the atomically
finalized `rendered/` root, never on the render staging directory. Backed by a
**negative test matrix**
(delete / add / rename / wrong-dtype / wrong-shape / stale-file / missing-file /
index-discontinuity — not only scalar flips).

Currently-missing checks to add (separately tested): render modes, principal
point, horizontal/vertical FOV, forward-mask radius, depth-scale agreement,
canonical depth ↔ alias, per-episode summary counts (depth only — RGB keeps
`episodes=[]`).

`np.load(..., allow_pickle=False)` in a context manager; validate keys, shapes,
dtypes, finiteness, common frame length.

### Float comparison policy (binding)

Use the narrowest comparison that matches how each artifact is authored:

1. **Summary ↔ summary / canonical depth ↔ alias:** exact parsed value equality
   (including floats) because both files are authored from the same object.
2. **Manifest camera height ↔ NPZ `camera_positions[:, 2]`:** cast the
   authoritative manifest value with `np.float32` and require exact equality to
   every stored float32 Z value.
3. **Summary/manifest calibration ↔ Parquet float32:** construct the expected
   intrinsic, extrinsic, and distortion arrays, cast them to `np.float32`, and
   compare exactly to the decoded Arrow float32 values.
4. **Isaac calibration readback:** preserve `rtol=1e-6`, `atol=1e-6` against
   requested runtime values.
5. **Legacy CLI expected-value assertions:** width/height compare exactly;
   horizontal FOV, fisheye coefficients, and camera height use
   `rtol=1e-6`, `atol=1e-6` against their authoritative summary/manifest field.
   A supplied mismatch raises before staging/output writes; omission disables
   only that assertion and never changes the authoritative value.

Tests cover each category, both just-inside and just-outside tolerant bounds,
and specifically `0.6` JSON/Python values versus stored float32 representations.

---

## Calibration authority (specified)
1. Render authors calibration; writes it into its summary.
2. `validate_pipeline_contract` checks RGB ↔ canonical-depth calibration agreement.
3. `package` constructs `CameraCalibration` from the canonical depth summary;
   derives `info.json` + parquet intrinsic/distortion from it.
4. Legacy package camera CLI fields = optional expected-values (individually
   optional). Width/height/FOV/coefficients must match canonical depth summary
   fields; `--camera-height` must match manifest `camera_height_m`. All supplied
   mismatches raise before writes under the binding float policy above.
   `--camera-height` defaults to `None`; manifest height remains authoritative.
5. `CameraCalibration.extrinsic_matrix(height)` delegates to
   `frames.camera_extrinsic` (single impl) — or the method is removed.
6. `CameraCalibration.from_cli(...)` still needed by render (the author); not by
   package after Phase 2a.
7. Isaac runtime **calibration readback** (`camera.get_opencv_fisheye_properties()`)
   preserved.

### Packaging compatibility policy (decided)

This refactor is **behavior-preserving**, not a LeRobot format migration. Keep
the current parquet, JSON/JSONL, and individual JPEG/PNG inventory exactly.
The current layout is **project-specific LeRobot-v2.1-style metadata**: images
live under `videos/`, while `info["video_path"]` advertises MP4 paths and the
image streams are absent from `features`. Standard LeRobot v2.1 loading is not
claimed by this refactor.

Phase 0a records any actual in-repo or external downstream consumer that is
available; the package oracle is the binding compatibility contract when no
such consumer exists. Phase 5 adds a checked-in format note and a smoke test for
the named consumer when one is found. Converting to standard LeRobot is a
separate post-refactor migration with its own output-version boundary and data
regeneration plan. The existing `codebase_version` value remains unchanged for
artifact compatibility and is not treated as a standards-compliance claim.

### Depth-encoding authority (decided)

The canonical depth summary owns `depth_type`, `min_depth_m`, `max_depth_m`, and
`depth_scale`. Package derives its depth metadata from those fields. The default
scale preserves the exact legacy string `uint16_meters_x_10000`; a non-default
scale must produce `uint16_meters_x_<scale>`, with `<scale>` formatted by
`np.format_float_positional(float(scale), unique=True, trim="-")`, rather than
silently claiming ×10000. RGB/depth summaries must agree on shared depth
settings even though RGB does not encode depth. Test at least `10000.0` (legacy
string unchanged), `5000.0`, and one non-integral valid scale.

---

## Output-state policy (chosen; whole-root render publication)

**Producers are non-destructive** (chosen over a `replace_existing` flag — safer
for library callers): final publication targets must be **absent**; producers
never delete existing final output. **Only the shell owns destructive
replacement** (`--force` removes the target before invoking the producer).
Direct CLI/library callers who want replacement must clear the target
themselves. No `replace_existing`/`--force` config field or CLI option is added
to producers.

All three producers create staging directories as **siblings of their own final
target** and `publication.py` verifies the staging and target parent device IDs
before rename. Cross-filesystem copy fallback is forbidden because it would
weaken atomicity; fail with an actionable error instead. Concretely, package
staging for an `/ssd5/.../<scene>` target also lives under that `/ssd5` parent,
never under the shell's `/tmp` work root.

Publication supports **one cooperative publisher per target**. It rechecks
target absence immediately before `os.rename` and tests both empty and non-empty
existing targets, but does not claim race-safe no-clobber behavior against a
non-cooperating process that creates the target between check and rename.
Concurrent publication to the same target is explicitly outside the contract;
do not add implicit replacement, copy fallback, or a portability claim for
`renameat2(RENAME_NOREPLACE)`.

- **Generate (`write_trajectory_artifacts`):** stage to a sibling dir → validate
  → atomic rename into the absent target.
- **Package (`package`):** stage the complete dataset in a sibling directory →
  atomic rename; **shell no longer pre-creates the final output dir**.
- **Render (two processes, one staged root):** RGB and depth both write into one
  sibling staging directory, e.g. `.rendered.<run-id>.staging`; they never write
  into the final `rendered/` target. A modality may create the staging root or
  consume a root containing only the other completed modality, but its own image
  directory and summary files must be absent. Before writing, a strict preflight
  accepts only an absent root or the exact, fully valid inventory of the other
  modality; partial, unrelated, or same-modality entries are rejected. After
  both processes finish,
  `cli/finalize_render.py` runs the complete render/trajectory contract against
  the staging root and atomically renames the **entire directory** to the absent
  `rendered/` target. The staging and target paths must be siblings on the same
  filesystem.
- **Accepted render-stage states:** a mode may start from an absent/empty root,
  complete RGB-only inventory, or complete depth-only inventory. It may add only
  the missing modality. Finalization accepts only the exact complete
  two-modality inventory. Any partial modality, duplicate own modality,
  unrelated path, or summary/image mismatch is rejected.
- **Failure/restart semantics:** an exception or process death may leave an
  incomplete staging directory, but the final target remains absent. Producers
  never clean or reuse ambiguous partial staging state. The shell starts with a
  new staging path (and may remove an old one only under its explicit
  replacement ownership); direct callers must inspect/remove the partial stage
  or choose a new stage. Tests cover exceptions, process-restart fixtures with
  pre-seeded partial states, refusal to overwrite a modality within staging, and
  final-target non-overwrite.

---

## Final target tree (transitional; includes rollout shims)

```
sage3d/
  __init__.py              # empty
  config.py                # [package-safe: stdlib+pathlib only] SceneConfig,
                           #   SafetyConfig, PathConfig, GenerationConfig,
                           #   PackageConfig, RenderConfig (mode as str/Literal)
  artifacts.py             # [stdlib+pathlib only] resolve_generation_assets,
                           #   resolve_render_assets (stage-specific)
  publication.py           # [stdlib+pathlib only] absent-target checks +
                           #   same-filesystem atomic directory publication
  frames.py                # [numpy only] yaw_to_quaternion, camera_extrinsic,
                           #   yaw_to_rotation2d, COORDINATE_FRAME
  camera.py                # [numpy only] CameraCalibration (wraps fisheye_camera)
  episode_arrays.py        # [numpy ONLY] EpisodeArrays npz schema (save/load)
  naming.py                # [stdlib only] episode_filename, parse_episode_filename,
                           #   frame_stem, parse_frame_filename (explicit int IDs)
  io_ply.py                # [numpy only] write_binary_pointcloud +
                           #   read_binary_pointcloud_metadata (writer+parser)
  pointcloud.py            # [numpy only] voxel_downsample
  schemas.py               # [numpy only] TrajectoryManifest, RenderSummary
                           #   (mode-aware), TrajectoryEpisodeRecord
  contract.py              # [numpy+PIL] validate_pipeline_contract (explicit
                           #   exceptions; pre-package only)
  geometry.py              # [cv2/numpy] MapTransform, pixels_to_world,
                           #   path_length, wrap_angle  (MapInfo stays a dict until 3b)
  pathfinding.py           # [numpy] NEIGHBORS, astar
  path_postprocess.py      # [scipy/numpy] points_are_safe, simplify, smooth, resample
  navigation_map.py        # [cv2/numpy/PIL] load_navigation_map,
                           #   connected_components, MapInfo (introduced in 3b)
  collision.py             # [trimesh/pxr] extract_collision_geometry,
                           #   collision_distances (batched), apply_camera_clearance
  viz.py                   # [cv2/PIL] save_navigation_visualizations
  episode_generation.py    # [scipy/trimesh/cv2 — Isaac side] EpisodeCandidate,
                           #   generate_episodes, build_episode_arrays
  trajectory_pipeline.py   # generate(config)->TrajectoryResult ;
                           #   write_trajectory_artifacts (staging+rename)
  render_processing.py     # [numpy only — no Isaac] build_forward_mask, mask_rgb,
                           #   encode_depth, RawDepthSummaryAccumulator
  render_bootstrap.py      # [Isaac] pure imports; parse config, start SimulationApp,
                           #   import render_runtime, close in finally
  render_runtime.py        # [Isaac] imported AFTER app: RenderMode strategy,
                           #   render_episode, render(config, staging_root)
  lerobot_dataset.py       # [pyarrow] build_episode_parquet, copy_episode_frames,
                           #   write_lerobot_meta, package (staging+rename)
  cli/
    __init__.py            # package marker (explicit; no namespace-package reliance)
    _args.py               # add_fisheye_args, add_scene_args
    generate.py            # Phase 3e
    render.py              # Phase 4
    finalize_render.py     # Phase 4; validate staging then atomic publish
    package.py             # Phase 5
fisheye_camera.py          # unchanged (shared with other datasets)
# rollout compatibility shims (removal gate defined in Phase 6):
generate_sage3d_trajectories.py   # -> sage3d.cli.generate
render_fisheye_sage3d.py          # -> sage3d.cli.render
package_lerobot_sage3d.py         # -> sage3d.cli.package
run_pipeline_sage3d.sh            # SCRIPT_DIR-derived; explicit PYTHONPATH so
                                 #   `python -m sage3d.cli.*` resolves from any CWD
tests/package_safe/               # package-python unit/contract tests
tests/isaac/                      # Isaac-python generation/runtime tests
tests/integration/                # subprocess + end-to-end tests
tests/golden/<scene>/             # Phase 0b pinned characterization (see storage)
scripts/check_{generate,package,render}.py   # Phase 0b (checker scripts; not in the
                                             #   module-creation audit, which covers
                                             #   production modules + config only)
```

**Module discovery (decided):** do not install the package and do not depend on
the caller's CWD. Every shell/subprocess invocation prepends the repository's
`SCRIPT_DIR` to the inherited `PYTHONPATH`:

```bash
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$SAGE3D_ISAAC_PYTHON" -m sage3d.cli.generate ...
```

Use the same form with `SAGE3D_PACKAGE_PYTHON`; this also keeps the unchanged
root-level `fisheye_camera.py` importable. No `pyproject.toml`/installation work
is in scope. The executable
`python -m sage3d.cli.generate|render|finalize_render|package --help` smoke tests
land in Phases 3e / 4 / 4 / 5 respectively; Phase 2c imports only then-existing
modules from outside the repo under both interpreters.

### Render/finalizer CLI and compatibility contract (decided)

The new pipeline uses this exact boundary (camera arguments repeated for both
render processes; shown explicitly to avoid implicit shared state):

```bash
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$SAGE3D_ISAAC_PYTHON" -m sage3d.cli.render \
  --mode rgb --scene "$SCENE" --sage-root "$SAGE_ROOT" \
  --trajectory-dir "$TRAJECTORY_DIR" --staging-root "$RENDER_STAGE" \
  --width "$WIDTH" --height "$HEIGHT" \
  --horizontal-fov-deg "$HORIZONTAL_FOV_DEG" \
  --fisheye-coefficients "${FISHEYE_COEFFICIENTS[@]}"

PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$SAGE3D_ISAAC_PYTHON" -m sage3d.cli.render \
  --mode depth --scene "$SCENE" --sage-root "$SAGE_ROOT" \
  --trajectory-dir "$TRAJECTORY_DIR" --staging-root "$RENDER_STAGE" \
  --width "$WIDTH" --height "$HEIGHT" \
  --horizontal-fov-deg "$HORIZONTAL_FOV_DEG" \
  --fisheye-coefficients "${FISHEYE_COEFFICIENTS[@]}"

PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$SAGE3D_PACKAGE_PYTHON" -m sage3d.cli.finalize_render \
  --scene "$SCENE" --trajectory-dir "$TRAJECTORY_DIR" \
  --staging-root "$RENDER_STAGE" --output-dir "$RENDERED_DIR"
```

`sage3d.cli.render` accepts `--staging-root` and does **not** accept
`--output-dir`; `finalize_render` alone accepts both staging and final output
paths. The legacy `render_fisheye_sage3d.py` shim retains its old `--output-dir`
surface by mapping that path to the new renderer's staging root, emits an
actionable deprecation/non-atomic warning, and never invokes the finalizer.
Thus two direct legacy invocations preserve the legacy artifact inventory but
are an explicit non-atomic compatibility exception; only the module-CLI +
finalizer sequence claims validated atomic publication. Automatic “second mode
finalizes” behavior is forbidden.

**Golden storage/provenance:** before formal handoff, commit this plan and record
the repository commit in the Phase 0b provenance. In git: provenance manifest,
input-asset SHA-256 values, exact commands/config, expected inventory, selected
small RGB/depth golden frames/metrics, and all pinned tolerances. The provenance
also records OS/kernel; both Python versions; Isaac Sim/Kit/USD; GPU model,
driver, CUDA/runtime and renderer settings; and exact NumPy, SciPy, trimesh,
rtree, OpenCV, Pillow, and PyArrow versions in the environment where each stage
runs. Outside git: the full rendered dataset, referenced by immutable URI/cache
key and checksum from the provenance record.

The canonical GPU lane automatically fingerprints its hardware/runtime and must
match the Phase 0b provenance. On a nonmatching GPU, structural/logic checks and
synthetic encoding tests remain binding, but RGB/depth golden-frame thresholds
are report-only and cannot replace canonical PR evidence. Do not trust a
manually supplied fingerprint as the source of truth.

**Checker execution contract:** all `scripts/check_*.py` programs are read-only,
run under package Python, take actual artifact path + baseline/provenance path,
verify the applicable recorded runtime/library fingerprint before applying a
version-sensitive golden assertion, emit a concise human/JSON result, and exit
nonzero on binding failure.
`check_generate.py` implements the complete generation oracle;
`check_render.py` implements static inventory/calibration/depth checks plus the
matching-GPU golden policies; `check_package.py` implements semantic Arrow,
JSON/JSONL, checksum/inventory, and post-package float32 calibration/extrinsic
checks. After a producer is rewired, its Isaac/GPU lane creates **fresh**
artifacts and the package-safe checker compares those artifacts to the pinned
baseline; merely rechecking the old snapshot does not satisfy a phase DoD.

---

## Implementation phases

Each phase is independently mergeable. Tests for a new API **co-land with the
phase that creates it**; DoDs import only modules that exist at that phase.

### Decisions recorded before Phase 0a (so an engineer can start tomorrow)
1. **Three validation lanes:**
   - package-safe: `$SAGE3D_PACKAGE_PYTHON -m pytest tests/package_safe`;
   - Isaac-side: `$SAGE3D_ISAAC_PYTHON -m pytest tests/isaac`;
   - pinned GPU/integration: `$SAGE3D_ISAAC_PYTHON -m pytest -m sage3d_gpu tests/integration`,
     plus package-python checker subprocesses where needed.
   The package runner never imports Isaac-side modules. Test dependencies and
   pytest marker registration are documented/checked in during Phase 0a.
2. **Interpreter paths via env vars** `SAGE3D_ISAAC_PYTHON` /
   `SAGE3D_PACKAGE_PYTHON` — no hardcoded `/ssd4/...` in tests.
3. **Phase 0a scope:** `fisheye_camera.py` unit tests + synthetic package
   success fixture + artifact parsers that import no target modules + an
   inventory of actual downstream readers/callers (record “none found” when
   applicable). **Live generate/render characterization is Phase 0b only.**
4. **Assets:** synthetic fixtures for 0a; external SAGE3D assets only for 0b.
5. **Skip/gate policy:** developer/general CI runs may skip tests whose declared
   Isaac/GPU/assets prerequisites are unavailable; the canonical `sage3d_gpu`
   validation lane treats missing prerequisites as a **failure**, not a skip.
   If no hosted GPU CI exists, its command output + provenance are attached as
   required PR evidence. Phase 0a green requires only no-asset tests; every
   later DoD that says “Phase 0b checks pass” requires evidence from the
   canonical lane.
6. **Incremental forbidden-import module list** as above.
7. RGB tolerances (T₁, T₂, N) and selected-depth mask/error-percentile
   tolerances established from the Phase 0b baseline.
8. **Handoff identity:** revision 5 is committed before baseline capture; every
   provenance/checker record names that commit rather than an untracked file.

### Phase 0a — Legacy characterization tests (no target modules)  *(~1 day)*
Per the decisions above. **DoD:** green on the unmodified codebase (no-asset
tests) under package python. **Risk:** low.

### Phase 0b — Pinned external baselines + artifact checkers  *(~1–2 days)*
First run a generation-only feasibility smoke using scene `839920`, seed
`20260720`, five episodes, and the current generation defaults; stop before the
expensive render baseline if it cannot complete within `max_attempts=3000`.
Then run the pinned generate/render/package golden pipeline with the current
default camera and full provenance; implement `check_{generate,package,render}.py`;
derive RGB T₁/T₂/N and selected-depth mask/error tolerances from the baseline.
Changing scene/seed/count requires an explicit plan/provenance update, not an
implementer-local substitution. **Formal exit gate before Phase 1:** all three
checkers reproduce from provenance; required provenance fields are complete;
the canonical GPU fingerprint is captured; and the owner accepts a revised
schedule/risk estimate based on measured runtime, retry rate, artifact size,
and checker complexity. **Risk:** medium until the gate closes.

### Phase 1 — Package-safe leaf modules + wiring  *(~1.5 days)*
**Create:** `__init__.py`, `frames.py`, `camera.py` (+ width/height/FOV and
finite-value validation), `episode_arrays.py`, `naming.py`, `io_ply.py` (writer + parser),
`pointcloud.py` (`voxel_downsample`), `publication.py` (absent-target and
same-filesystem atomic-directory helpers), `render_processing.py`
(`build_forward_mask`, `mask_rgb`, `encode_depth`,
`RawDepthSummaryAccumulator` with the contracts above), `cli/__init__.py`,
`cli/_args.py` (`add_fisheye_args`).
**Wiring (behavior-preserving; CLI surfaces unchanged):** render uses
`frames`+`CameraCalibration`+`naming`+`render_processing` (preserve Isaac
readback); package uses `CameraCalibration`+`naming`; generate writes npz via
`EpisodeArrays`, uses `frames.yaw_to_rotation2d`+`pointcloud`+`io_ply`+`naming`.
**DoD:** Phase 0a/0b checks pass; **forbidden-import smoke test (Phase 1 module
list only)** green in fresh subprocess; `check_render.py` green; unit tests for
frames/camera/naming/io_ply/pointcloud/publication/render_processing green.
**Risk:** low–medium.

### Phase 2a — Typed schemas + calibration authority  *(~1 day)*
**Create:** `schemas.py` (`TrajectoryManifest`, `TrajectoryEpisodeRecord` — the
manifest episode type from `serializable_episode`; mode-aware `RenderSummary`),
`config.py` (empty package-safe shell). **Modify:** generate writes manifest via
`TrajectoryManifest.to_json`; render writes summary via `RenderSummary.to_json`;
package constructs `CameraCalibration` from canonical depth summary, derives
info/parquet from it, legacy camera CLI = optional expected-values, reads
`camera_height` from manifest. **JSON equality:** parsed semantic equality +
explicit key-order comparison. **Note:** the LeRobot-style `meta/episodes.jsonl`
record (Phase 5) is a **distinct** shape, not `TrajectoryEpisodeRecord`.
**DoD:** Phase 0b checks pass; manifest/summary content unchanged; forbidden-import
adds `config`,`schemas`; `check_render.py` green. **Risk:** medium.

### Phase 2b — Cross-artifact validator + negative matrix  *(~1 day)*
**Create:** `contract.py::validate_pipeline_contract` (signature above; explicit
exceptions; **pre-package invariants only**) + full negative test matrix.
**Modify:** `package` calls it. **DoD:** negative matrix green; Phase 0b checks
pass; forbidden-import adds `contract`. **Risk:** medium.

### Phase 2c — Asset resolver + `add_scene_args` + module discovery  *(~0.5 day)*
**Create:** `artifacts.py` with stage-specific `resolve_generation_assets(...)` /
`resolve_render_assets(...)` (generation needs **no USDZ**; render needs **no
InteriorGS dir**); lazy/`require_*` validation preserving `resolve_scene_dir()`
exactly-one-match; `cli/_args.add_scene_args`. **Modify:** **both** generate and
render accept `--sage-root`+`--scene`; `--interiorgs-root`/`--usdz`/`--collision-usd`
become higher-priority overrides; shell derives `SCRIPT_DIR` + uses
the binding repository-prefixed `PYTHONPATH` invocation only (no `cd` or
installation alternative). **DoD:** Phase 0b checks pass; resolver tests cover
zero/multiple matches, file-vs-dir predicates, all partial-override combos,
override precedence, legacy commands, generation-not-requiring-USDZ;
**discovery smoke test imports only existing modules** from outside the repo
under both interpreters (executable `python -m sage3d.cli.*` tests deferred to
3e/4/5). **Risk:** low.

### Phase 3a — Exact generation leaf extraction  *(~1.5 days)*
**Verbatim moves** (no algorithmic change; `MapInfo` stays the current **dict**;
`cumulative_distances` **not** introduced here):
- `geometry.py` ← `MapTransform`, `pixels_to_world`, `path_length`, `wrap_angle`;
- `pathfinding.py` ← `NEIGHBORS`, `astar`;
- `path_postprocess.py` ← `points_are_safe`, `simplify_by_visibility`,
  `smooth_path`, `resample_path` (scalar `points_are_safe`);
- `collision.py` ← `extract_collision_geometry` (drop unused `Gf`),
  `collision_distances` (preserve `2048`), `apply_camera_clearance`;
- `viz.py` ← `save_navigation_visualizations`;
- `navigation_map.py` ← `load_navigation_map`, `connected_components`.
**DoD:** full generation oracle (`np.array_equal` + manifest incl key-order +
exact path-field normalization for `scene_dir`/`collision_usd` + **exact PLY
bytes** + decoded-viz equality/checksums). **Risk:** low.

### Phase 3b — Generation config + `MapInfo` + exact `episode_generation.py`  *(~1 day)*
**Create:** `config.py` adds `SceneConfig`/`SafetyConfig`/`PathConfig`/
`GenerationConfig` (validations: episode count, attempts, path ranges, frame
spacing, seed policy, radius/margins, camera height/clearance, endpoint
clearance, voxel size, max points); `navigation_map.MapInfo` dataclass replaces
cross-call-site `map_info` mutation with one final construction **preserving
serialized field values + key order**; `episode_generation.py` ← **exact** move
of `generate_episodes` + `build_episode_arrays`. **DoD:** full generation oracle
green. **Risk:** low–medium.

### Phase 3c — Decompose the rejection loop  *(~1.5 days)*
Inside `episode_generation.py`: `EpisodeCandidate` + rejection-result types;
split `generate_episodes` into `sample_endpoint_pair` → `plan_path` →
`postprocess_path` → `validate_camera_clearance` → `build_episode`. **Preserve
exactly:** RNG draw order, rejection-check order, attempt counting, dict +
episode order, dtypes + operation order, smoothing strategy order + labels; fix
`used_endpoints[0::2]`/`[1::2]` → `list[tuple[start, goal]]` result-preserving.
Add deterministic call/rejection traces covering all rejection exits.
**DoD:** full generation oracle (`np.array_equal`) green. **Risk:** medium.

### Phase 3d — `trajectory_pipeline` + generation output-state  *(~1 day)*
**Create:** `trajectory_pipeline.py` — `TrajectoryResult`, `generate(config)`
(orchestration **separated from writes**; not "pure"), `write_trajectory_artifacts`
(staging + atomic rename; target-absent precondition; non-destructive).
**DoD:** full generation oracle green; failure-injection + rerun for sibling
staging; same-device assertion; existing empty/non-empty target rejection; and
single-publisher/no-concurrency contract tests. **Risk:** low–medium.

### Phase 3e — Thin generation CLI + compat shim  *(~0.5 day)*
**Create:** `cli/generate.py`. `generate_sage3d_trajectories.py` → rollout
shim. Shell switches generate to `python -m sage3d.cli.generate`, creates only
`WORK_DIR`, and **removes the legacy `mkdir -p "$TRAJECTORY_DIR"`** so the final
generation target is absent. **DoD:** a shell/subprocess test observes an absent
trajectory target at CLI entry; `python -m sage3d.cli.generate --help` works
from outside the repo; exit code + artifacts match legacy. **Risk:** low.

### Phase 4 — Render extraction + atomic whole-root finalization  *(~2.5 days)*
**Create:** `config.RenderConfig` (width/height, startup/settle steps ≥ 0,
finite positive depth scale, `0 < min < max`, `max_depth_m*depth_scale ≤ 65535`;
**package-safe**); `render_bootstrap.py` (pure imports → construct
`SimulationApp` → import `render_runtime` → run → **close in `finally`** incl
failures during runtime imports/stage setup); `render_runtime.py` (`RenderMode`
strategy: `.build_stage`/`.configure_camera`/`.capture`, single `render_episode`
loop using `render_processing`, `render(config, staging_root)`); `cli/render.py`;
`cli/finalize_render.py` (load staged artifacts → run the full contract → call
`publication.atomic_publish_directory`). Keep the **two-process** model (a mode
does not "spawn" an app). `render_fisheye_sage3d.py` → rollout shim. The shell
runs RGB and depth against the same sibling staging root, then invokes the
exact finalizer command under `SAGE3D_PACKAGE_PYTHON`; new CLI and legacy-shim
behavior follow the binding contract above. **DoD:** render oracle green (all
10 items incl exact `encode_depth`, selected golden depth, streaming accumulator,
mocked stage construction with
**exact reference string incl `[gauss.usda]`**, the complete warmup/capture call
trace, pre-encode mask, leakage T₁/N, golden T₂, and app-close-on-failure);
failure injection before/within each modality and immediately before final
publication leaves the final target absent; pre-seeded partial-stage and
same-modality-overwrite tests green; finalizer rejects incomplete/stale/invalid
inventories and an existing final target; `python -m sage3d.cli.render --help`
and `python -m sage3d.cli.finalize_render --help` work outside the repo.
**Risk:** medium–high until logic/sequencing/stage/publication tests land.

### Phase 5 — Package extraction + explicit format contract  *(~2 days)*
**Create:** `config.PackageConfig` (positive FPS, path/output requirements,
optional compatibility-assertion fields); `lerobot_dataset.py`
(`build_episode_parquet`, `copy_episode_frames`, `write_lerobot_meta`, `package`
with sibling staging + atomic rename; non-destructive); `cli/package.py`; a
checked-in format note documenting the project-specific LeRobot-style layout
and the separate standardization follow-up. Package consumes only the finalized
render root, constructs calibration and depth metadata from the canonical depth
summary, and preserves the default baseline output exactly. Add the
non-default-depth-scale test plus the complete float-policy positive/negative
matrix. The shell no longer pre-creates final output;
`--force` removes it before invoking package and switches to
`python -m sage3d.cli.package`. `package_lerobot_sage3d.py` → rollout shim.
If Phase 0a found a real downstream consumer, its pinned smoke test is required;
otherwise the semantic package oracle is binding and the lack of a named
consumer is recorded, not left as an implementer choice. Run
`prepare_trajectories.py` as a supplemental smoke for shared Parquet path,
`observation.camera_extrinsic`, and `action` only; failure is actionable, but it
does not replace the SAGE3D package oracle. **DoD:** package oracle
green **including post-package extrinsic/calibration/depth-metadata checks in
`check_package.py`**; exact inventory and default metadata preserved;
non-default depth scale truthful; failure-injection, partial-staging, rerun, and
existing-target refusal tests green; `python -m sage3d.cli.package --help`
works outside the repo. **Risk:** low–medium.

### Phase 6 — Integration, portability, documentation, rollout  *(~1 day)*
Finalize `run_pipeline_sage3d.sh`: derive `SCRIPT_DIR`; honor
`SAGE3D_ISAAC_PYTHON`/`SAGE3D_PACKAGE_PYTHON` with documented local defaults;
prepend the binding `PYTHONPATH`; create every staging directory beside its own
final target; run RGB → depth → package-Python finalizer → package; never
pre-create publication targets; and confine destructive cleanup to explicit
shell-owned `--force`/work-directory operations. Preserve `--plan-only` as the
generate-only early exit. Replace the current inline frame/parquet inventory
validator with authoritative `check_package.py` invocation (its concise success
summary replaces the old counts; do not maintain duplicate validation logic).
Update README with
module CLI examples, environment setup, project-specific package compatibility,
staging/recovery rules, and output authority. Run the pinned full pipeline from
a CWD outside the repo and archive validation evidence. **DoD:** all three test
lanes and all checker scripts green; legacy shell and shim invocations preserve
exit codes/default artifacts apart from documented safer existing-output
refusal and deprecation notices; no hardcoded repository path remains.

Compatibility shims remain after Phase 6 and emit an actionable deprecation
notice. Removal is a separate PR allowed only after: (a) repository references
and docs use module CLIs, (b) known downstream callers are inventoried and
signed off, and (c) at least one complete dataset-regeneration rollout has used
the new entry points. This replaces the ambiguous “one release” rule.
**Risk:** medium because it is the cross-environment integration gate.

### Phase 7 — Optional performance work  *(profile-driven, one PR per change)*
Each PR defines, up front: benchmark workload (the pinned scene full pipeline),
**minimum improvement** (e.g. ≥ 1.5× wall-clock on the workload, else reject),
**max memory regression** (e.g. ≤ 1.2×), **trajectory-compatibility rule per
item** (vectorized `points_are_safe` ⇒ trajectories **may change** ⇒ new
data-contract version boundary + explicit acceptance; vectorized pixel↔world with
unchanged rounding ⇒ bit-identical output; cached proximity ⇒ identical output;
PLY structured-dtype ⇒ exact bytes), and a version-boundary policy. Items:
vectorized `points_are_safe` (**riskiest** — rejection-path), vectorized
pixel↔world (unchanged rounding only), vectorized minimum-clearance lookup,
cached `trimesh` proximity (preserve batching), PLY structured-dtype
serialization (unaligned 15-byte LE record), any batched/parallel packaging.

---

## Module / config creation audit (every item assigned once; production modules + config only)

| Item | Created in | Env | Notes |
| --- | --- | --- | --- |
| `__init__.py` | 1 | pkg | empty |
| `frames.py` | 1 | pkg | |
| `camera.py` (`CameraCalibration`) | 1 | pkg | extrinsic delegates to `frames` |
| `episode_arrays.py` | 1 | pkg | numpy only |
| `naming.py` | 1 | pkg | stdlib only |
| `io_ply.py` (writer+parser) | 1 | pkg | numpy only |
| `pointcloud.py` (`voxel_downsample`) | 1 | pkg | numpy only |
| `publication.py` | 1 | pkg | stdlib+pathlib; absent-target + atomic directory publish |
| `render_processing.py` | 1 | pkg | owns `encode_depth` + streaming `RawDepthSummaryAccumulator`; consumed by `render_runtime` in 4 |
| `cli/__init__.py`, `cli/_args.add_fisheye_args` | 1 | pkg | |
| `schemas.py` | 2a | pkg | `TrajectoryEpisodeRecord` = manifest episode type |
| `config.py` (shell) | 2a | pkg | package-safe |
| `contract.py` | 2b | pkg (numpy+PIL) | pre-package invariants only |
| `artifacts.py`, `cli/_args.add_scene_args` | 2c | pkg | stdlib+pathlib; stage-specific resolvers |
| `geometry.py` | 3a | Isaac | incl `pixels_to_world`, `path_length` |
| `pathfinding.py` (NEIGHBORS, astar) | 3a | Isaac | |
| `path_postprocess.py` | 3a | Isaac | scalar `points_are_safe` |
| `collision.py` (extract, collision_distances, apply_camera_clearance) | 3a | Isaac | drop unused `Gf`; keep `2048` |
| `viz.py` | 3a | Isaac | |
| `navigation_map.py` (load, components, **dict**) | 3a | Isaac | current load path uses PIL in addition to cv2/numpy; `MapInfo` class deferred to 3b |
| `Scene/Safety/Path/GenerationConfig`, `MapInfo` | 3b | pkg / Isaac | config pkg-safe; MapInfo Isaac-side |
| `episode_generation.py` (exact move) | 3b | Isaac | |
| `EpisodeCandidate` + rejection decomposition | 3c | Isaac | |
| `trajectory_pipeline.py` | 3d | Isaac | staging+rename; non-destructive |
| `cli/generate.py` | 3e | Isaac | rollout shim retained through migration gate |
| `RenderConfig` (in `config.py`) | 4 | **pkg-safe** | mode is str/Literal |
| `render_bootstrap.py`, `render_runtime.py`, `cli/render.py` | 4 | Isaac | bootstrap/runtime split |
| `cli/finalize_render.py` | 4 | pkg | validate complete staging root, then publish |
| `PackageConfig`, `lerobot_dataset.py`, `cli/package.py` | 5 | pkg | project-specific format contract |
| Shell/docs rollout | 6 | mixed | portability, recovery, deprecation, full integration |
| Phase 7 perf items | 7 | varies | one PR each, with acceptance criteria |
| shim removal | post-rollout | — | separate PR after the Phase 6 removal gate |

> Scope: this table covers **production package modules + config classes only**.
> `scripts/check_*.py` (Phase 0b) and `tests/` helpers are not production modules.

---

## Risk assessment
- **0a:** low (read-only inventory and characterization).
- **0b:** medium and the largest schedule uncertainty (large artifacts, asset
  availability, canonical-GPU access, scene feasibility, runtime/version drift,
  and tolerance characterization). The feasibility smoke and formal exit gate
  bound this risk before Phase 1.
- **2a:** medium (calibration authority + JSON key-order).
- **2b:** medium (safety-critical validation; negative matrix guards it).
- **3c:** medium (RNG/draw-order preservation).
- **4:** medium–high until logic/sequencing/stage/publication tests land.
- **5:** low–medium (literal extraction plus depth-metadata authority).
- **6:** medium (cross-environment integration, portability, and rollout).
- **7:** behavior/perf changes; `points_are_safe` vectorization riskiest.

## PR sequencing (safe → risky)
0a → 0b → 1 → 2a → 2b → 2c → 3a → 3b → 3c → 3d → 3e → 4 → 5 → 6 → 7.

**Effort:** retain 15–19 engineer-days as the base implementation estimate
through Phase 6, but use **19–27 engineer-days as the delivery-planning range**
(25–40% contingency) until Phase 0b establishes baseline runtime, GPU
tolerances, and fixture stability. Both ranges exclude PR queue time, GPU
scheduling delays, and optional Phase 7 work. Re-estimation is a formal Phase
0b exit artifact accepted by the engineering owner; Phase 1 does not start
until it is recorded. Phases 0a–2c deliver the reusable contract foundation in
roughly 5–6 base days and can ship standalone; the primary
monolith/readability goal is not complete until Phases 3–5 land.

## Corrections logged across all five reviews
- Cumulative-distance idiom: **3** sites (not 4); `cumulative_distances`
  unification deferred out of "verbatim" Phase 3a.
- `2048`: collision-query batch (not pathfinding); preserved through refactor.
- `package` loads `render_summary.json`(depth)+`rgb_render_summary.json`;
  `depth_render_summary.json` is checked+copied only.
- Calibration recompute served validation **and** intrinsic/info.json building.
- Frame transforms are related but **not** duplicated operations.
- `generate(config)` = "orchestration separated from writes," not "pure."
- `Gf` unused — removed in `collision.py` extraction.
- `EpisodeArrays.to_json_dict()` dropped (manifest record ≠ npz payload).
- `episode_arrays.py` (pkg) split from `episode_generation.py` (Isaac).
- **Rev 2:** Phase 0a can't green-test not-yet-existing APIs → tests co-land with
  creating phases; 0a is legacy characterization only.
- **Rev 2:** render oracle had two invalid assertions (decoded-JPEG outside-mask
  "exactly black"; depth-summary recompute from PNG) — corrected.
- **Rev 2:** `config.py` entirely package-safe; `RenderConfig.mode` str/Literal.
- **Rev 2:** `python -m sage3d.cli.*` needs explicit discovery (PYTHONPATH/cd/install).
- **Rev 2:** added `naming.py`, `render_processing.py`, `cli/_args.py`,
  `io_ply` parser, `MapInfo`/`apply_camera_clearance`/`pixels_to_world`/
  `path_length`/`NEIGHBORS`/`EpisodeRecord` ownership; Phase 3 reordered.
- **Rev 2:** output-state policy chosen (staging + atomic rename).
- **Rev 2:** asset resolver stage-specific.
- **Rev 3:** forbidden-import smoke test made **incremental per phase** (was
  listing future modules in Phase 1's DoD).
- **Rev 3:** executable `python -m sage3d.cli.*` smoke tests **deferred to
  3e/4/5** (Phase 2c discovery test imports only existing modules).
- **Rev 3:** `artifacts.py` **added back to the target tree**; `RenderConfig`
  env label **corrected to pkg-safe**.
- **Rev 3:** packaged-extrinsic/calibration checks **moved to `check_package.py`**
  (pre-package validator can't inspect unwritten parquet).
- **Rev 3:** producers are **non-destructive** (no `replace_existing` field);
  shell owns `--force` removal.
- **Rev 3:** render staging = **commit-marker per modality** (multi-sibling
  publication can't be one atomic rename) with failure-injection points (a/b/c).
- **Rev 3:** render-oracle tolerances (T₁/T₂/N) **pinned from the Phase 0b
  baseline**; `summarize_raw_depth` contract specified; `encode_depth` raises on
  `uint16` overflow from Phase 1.
- **Rev 3:** `TrajectoryEpisodeRecord` (manifest) distinguished from the
  LeRobot-style `episodes.jsonl` record (then Phase 4; Phase 5 in rev 4).
- **Rev 3:** "Decisions recorded before Phase 0a" section added (runner,
  interpreter routing via env vars, scope, assets, skip policy, tolerances).
- **Rev 4:** standard LeRobot loading explicitly made a non-goal; current
  project-specific layout preserved and documented, with standardization split
  into a separate versioned migration.
- **Rev 4:** per-modality markers replaced by one sibling render staging root +
  full validation + atomic whole-directory publication; crash leftovers remain
  outside the final target and require explicit shell/direct-caller cleanup.
- **Rev 4:** package-safe, Isaac-side, and pinned-GPU test lanes defined;
  canonical GPU validation cannot silently skip missing prerequisites.
- **Rev 4:** `voxel_downsample` assigned to `pointcloud.py`; `encode_depth`
  assigned only to `render_processing.py`; `publication.py` added for shared
  absent-target/atomic-directory mechanics.
- **Rev 4:** canonical depth summary made authoritative for packaged depth
  metadata, including truthful non-default `depth_scale` handling.
- **Rev 4:** render oracle expanded to cover both global warmups, calibration
  readback/annotator ordering, and every per-episode first-frame warmup.
- **Rev 4:** render extraction moved before package extraction; Phase 6 added
  for shell portability, documentation, deprecation, and full integration;
  performance work moved to Phase 7.
- **Rev 5:** fixed the Phase 3 generation-publication/shell conflict: the shell
  creates only `WORK_DIR` and must leave `TRAJECTORY_DIR` absent.
- **Rev 5:** fully specified `encode_depth`, including invalid values, mask
  behavior, rounding, non-integral scale, overflow, mismatch failures, input
  non-mutation, and a selected golden depth-frame corruption check.
- **Rev 5:** replaced potentially materializing raw-depth summarization with a
  one-pass accumulator and defined empty/invalid-frame and shape/mask failures.
- **Rev 5:** made cross-artifact float comparisons explicit for JSON, float32
  NPZ/Parquet, Isaac readback, and legacy CLI expected-value assertions.
- **Rev 5:** pinned the render/finalizer CLI sequence, accepted staging states,
  package-Python finalization, and the deprecated legacy shim's non-atomic
  exception.
- **Rev 5:** chose repository-prefixed `PYTHONPATH` as the only module-discovery
  mechanism for this refactor; installation metadata remains out of scope.
- **Rev 5:** documented cooperative single-publisher semantics and the limit of
  `Path.rename()` no-clobber behavior, while requiring same-filesystem sibling
  staging for all producers.
- **Rev 5:** recorded the supplemental (non-authoritative)
  `prepare_trajectories.py` consumer smoke, corrected dependency annotations,
  expanded provenance, and made the post-Phase-0b re-estimate a formal gate.
