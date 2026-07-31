# SAGE3D Refactor Plan (revision 8)

Consolidate `generate_sage3d_trajectories.py`, `render_fisheye_sage3d.py`, and
`package_lerobot_sage3d.py` into a single `sage3d/` package, eliminating
cross-script duplication and making each stage independently testable.

> **Status:** plan only — no code written yet. Nine review passes incorporated
> into the frozen **revision 8 handoff candidate**. Revision 8 resolves staging
> ownership and default work-root creation, pins canonical digest framing and
> final verification evidence, and strengthens independent approval, tolerance
> detection budgets, characterization isolation, and mutation determinism.
> Architecture is frozen after this revision; Phase 0 findings update tickets
> or decision/provenance records unless they reveal contradictory factual
> evidence.

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
- **Phase 4 adds:** `sage3d.cli.create_staging`,
  `sage3d.cli.finalize_render`
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

## Oracles (precise; tolerances pinned by the Phase 0b capture protocol)

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

### Package validation and golden oracle

**Baseline-independent validation** applies to every package run, including
arbitrary scenes/seeds/episode counts/camera settings. Given the dataset,
trajectory, and finalized-render roots, validate: exact output inventory;
Arrow schema/column order/types/row and episode counts; parquet `action` and
`observation.point_goal` against the current NPZ arrays; parsed JSON/JSONL
internal consistency and record order; copied RGB/depth, pointcloud, manifest,
and summaries against the **current inputs** by checksum; no stale/extra files;
and all packaged-artifact calibration, extrinsic, and depth-metadata checks below.

**Canonical golden comparison** is an additional operation only for the pinned
Phase 0b configuration. Compare deterministic package-derived Arrow and
JSON/JSONL content semantically with the canonical baseline. Never require
fresh nondeterministic JPEG/PNG bytes or copied render summaries to equal the
package baseline: validate those copies against the current render inputs, and
delegate tolerant render regression to `check_render.py compare-golden`.

**Packaged-artifact extrinsic/calibration checks run both in production staging
validation and in `check_package.py`** (not in the source-only pipeline
validator; see Cross-artifact validation): manifest
`camera_height_m` ↔ every packaged parquet `observation.camera_extrinsic`
(shape `[4,4]`, identity rotation, `[2,3] == camera_height_m`, equal across
frames); render calibration ↔ packaged `info.json` camera model, resolution,
horizontal/vertical FOV, fisheye coefficients, pitch, and forward-mask radius;
render focal length/principal point ↔ parquet intrinsic; render distortion ↔
parquet distortion; manifest camera height ↔ `info.json` camera height and
parquet extrinsic. `features["observation.camera_intrinsic"]` is checked only as
the preserved dtype/shape schema declaration; `info.json` contains no intrinsic
matrix value, and this refactor must not add one.

### Render oracle (corrected, tolerances protocol-pinned)

Pre-encode + decoded split (decoded JPEG outside the mask is **not** exactly
black — lossy q95). Concrete rules, with numeric thresholds **established by
the Phase 0b repeated-run protocol**, not chosen by the Phase 4 implementer:
1. **Pre-encode RGB unit test:** `rgb[~mask] == 0` exactly.
2. **Decoded-JPEG mask leakage:** `rgb_mask_leakage_mean_max` is the maximum of
   the three per-channel mean normalized intensities outside the forward mask
   dilated by nonnegative integer `rgb_mask_dilation_pixels`; construct that
   NumPy-only mask from the same center with radius
   `forward_mask_radius_pixels + rgb_mask_dilation_pixels` (no SciPy morphology).
   It must be ≤ its pinned maximum.
3. **Golden RGB frames:** compare selected decoded JPEGs (not raw frames) in
   float64 normalized to `[0,1]`, over all pixels inside the original circular
   mask and all three channels. `rgb_masked_rmse` and
   `rgb_masked_abs_error_p99` must each be ≤ its separately pinned maximum.
   RMSE is `sqrt(mean(square(actual - baseline), dtype=float64))`; p99 is
   `np.percentile(abs(actual - baseline), 99, method="linear")` over the same
   flattened samples.
   SSIM is deliberately not used, avoiding an underspecified algorithm and a
   new SciPy/scikit-image dependency.
4. **`encode_depth` exact synthetic matrix:** NaN, ±inf, below/exactly-at min,
   ordinary valid, exactly-at/above max, inside/outside mask, `np.rint` half-step
   behavior, integral/non-integral scale, overflow, shape/mask mismatch, exact
   dtype/shape, and input non-mutation (contract below).
5. **Depth PNG structural checker:** inventory, shape, `uint16` dtype, encoded
   bounds, and outside-mask sentinel through the staged helper policy below
   (standalone saved helper in Phase 0b; production
   `render_processing.encoded_depth_sentinel` from Phase 1); **do not** compare
   encoded min/max against the raw summary's
   `finite_depth_min/max`.
6. **Selected golden depth frames:** within the circular mask, define non-max
   pixels using the shared encoded sentinel. Require
   `depth_non_max_mask_iou ≥ depth_non_max_mask_iou_min`; on the non-empty
   intersection of actual/baseline non-max pixels require `depth_error_p50`,
   `depth_error_p95`, and `depth_error_p99` (absolute encoded-unit errors) ≤
   their separately pinned maxima. IoU is integer intersection count divided by
   integer union count; an empty union or intersection fails. Percentiles use
   `np.percentile(..., method="linear")` on the intersection samples. These
   metrics must detect all-max or otherwise corrupted inside-mask output.
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
10. **Depth-overflow precondition:** in force from Phase 1 onward —
   `render_processing.encoded_depth_sentinel` casts/stores max depth as float32,
   multiplies and rounds through the same NumPy path as the encoder, and raises
   before `uint16` conversion if the finite scaled value exceeds `65535`.
   The render entry point calls it for **both RGB and depth modes** before
   launching `SimulationApp` and before creating/modifying any render output or
   staging artifacts; `encode_depth` calls it again before allocating output.
   `RenderConfig` performs stdlib-only basic range validation, while this helper
   is the precision-authoritative public guard. This is an intentional new guard
   for invalid legacy inputs, not a Phase 0b preservation expectation. Defaults
   remain `6.0 × 10000 = 60000`.
- `scripts/check_render.py` added in Phase 0b: `validate` runs in every
  render-touching phase (1, 2a, 2c, 4), and canonical evidence additionally
  runs `compare-golden`.

#### `encode_depth` contract (specified)

```python
encode_depth(
    depth: np.ndarray,
    circular_mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    depth_scale: float,
) -> np.ndarray  # shape == depth.shape, dtype == np.uint16

encoded_depth_sentinel(
    max_depth_m: float,
    depth_scale: float,
) -> np.uint16
```

- Require numeric two-dimensional `depth` and `circular_mask.dtype == np.bool_`
  with the exact same shape; reject mismatch before producing output. Process a
  private `np.float32` copy of depth to preserve current operation precision.
- `encoded_depth_sentinel` independently requires finite positive
  `max_depth_m`, finite positive `depth_scale`, a finite float32-cast max, and a
  finite scaled result. It then uses exactly
  `np.asarray([max_depth_m], dtype=np.float32) * depth_scale`, checks the finite
  scaled value against `65535` before conversion, then returns
  `np.rint(scaled).astype(np.uint16)[0]`. From Phase 1, both the encoder and
  structural checker call this production helper; no Python `round` or
  independent sentinel formula is permitted after the Phase 0b migration.
- Reject non-finite/non-positive scale, invalid depth range, or helper-detected
  overflow before mutating/allocating the result.
- Valid raw pixels = finite and `depth >= min_depth_m` and inside the mask.
- NaN, ±inf, below-min, and outside-mask pixels become `max_depth_m`.
- Valid values above max are clipped to `max_depth_m` for encoding only.
- Encode with `np.rint(encoded_meters * depth_scale).astype(np.uint16)`; preserve
  NumPy half-to-even behavior exactly.
- Never mutate `depth` or `circular_mask`.
- Pin `max_depth_m=1.001, depth_scale=500`: Python float64 `round` produces
  `500`, while the specified float32/NumPy path produces sentinel `501`. Also
  pin `max_depth_m=131.07, depth_scale=500`: the float64 product is `65535`,
  while the float32 product exceeds `65535` and must raise. These cases prevent
  checker, config, and encoder numeric paths from drifting.

#### Streaming raw-depth summary contract (specified)

Use one accumulator per episode; do not retain or materialize raw frames:

```python
summary = RawDepthSummaryAccumulator(circular_mask, min_depth_m)
for depth in depth_frames:
    summary.add(depth)
result: RawDepthEpisodeSummary = summary.finish()
```

To preserve current capture semantics without mutating the caller, `add` first
executes `frame = np.asarray(depth, dtype=np.float32).squeeze()`, then requires
a two-dimensional frame exactly matching the non-empty mask. It raises when the
frame contains no valid depth. Retain only the per-frame Python-float fraction,
minimum, and maximum in three ordered scalar lists—never the raw frame.
`finish` raises when zero frames were added and applies the legacy operation
order (`np.mean`/`np.min` over the fraction list and `min`/`max` over extrema
lists), rather than a differently ordered running sum. Preserve the current four
aggregate formulas exactly:
- per-frame valid mask = `np.isfinite(frame) & (frame >= min_depth_m) & circular_mask`;
- per-frame finite fraction = `valid_inside.sum() / circular_mask.sum()`;
- episode `finite_depth_fraction_mean`/`_min` = mean/min of **per-frame**
  fractions (not one global fraction — matters when frame masks vary);
- episode `finite_depth_min_m` = min over per-frame float32 minima;
  `finite_depth_max_m` = max over per-frame float32 maxima.
- Raw valid depths **above `max_depth_m` are included** in
  `finite_depth_max_m` (clipping is for PNG encoding only).
Tests cover float64 inputs straddling `min_depth_m` after float32 conversion,
singleton-dimension squeeze behavior, input non-mutation, and equality with the
legacy per-frame-list reduction order.

---

## Cross-artifact validation (preserved + strengthened; split corrected)

`contract.py::validate_pipeline_contract(...)` runs **pre-package** (in `package`
before any parquet is written) and therefore **cannot** inspect packaged
extrinsics/calibration — those are checked on the completed staging tree by
`validate_packaged_dataset` and independently by `check_package.py`. Explicit
signature:
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
extrinsic is checked on staging and after publication). Package invokes this
only on the atomically finalized `rendered/` root, never on the render staging
directory. Backed by a
**negative test matrix**
(delete / add / rename / wrong-dtype / wrong-shape / stale-file / missing-file /
index-discontinuity — not only scalar flips).

Currently-missing checks to add (separately tested): render modes, principal
point, horizontal/vertical FOV, forward-mask radius, and explicit RGB ↔ depth
agreement for **all shared depth fields** (`depth_type`, `min_depth_m`,
`max_depth_m`, `depth_scale`), plus canonical depth ↔ alias and per-episode
summary counts (depth only — RGB keeps `episodes=[]`).

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
   fisheye coefficients preserve the current packager's `np.allclose` policy,
   made explicit as `rtol=1e-5`, `atol=1e-8`; the derived focal-length
   compatibility check preserves `math.isclose(rel_tol=1e-6, abs_tol=0.0)` when
   the complete legacy calibration bundle is supplied. New direct horizontal
   FOV and camera-height assertions use `rtol=1e-6`, `atol=1e-6`. A supplied
   mismatch raises before staging/output writes; omission disables only that
   assertion and never changes the authoritative value. The new optional-field
   behavior is an intentional validation-policy change for inconsistent inputs;
   valid defaults and existing coefficient/focal tolerances are preserved.

Tests cover each category, both just-inside and just-outside tolerant bounds,
and specifically `0.6` JSON/Python values versus stored float32 representations.

---

## Calibration authority (specified)

1. Render authors calibration; writes it into its summary.
2. `validate_pipeline_contract` checks RGB ↔ canonical-depth calibration agreement.
3. `package` constructs `CameraCalibration` from the canonical depth summary;
   derives `info.json` camera metadata plus parquet intrinsic/distortion from it.
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

All three producers use staging directories that are **siblings of their own
final target**, and `publication.py` verifies the staging and target parent
device IDs before rename. Staging allocation ownership is explicit:

- Generate and package call
  `publication.create_staging_directory(final_target, prefix)` internally.
  The helper uses `tempfile.mkdtemp(dir=resolved_target_parent, ...)`.
- Render has two producer processes and therefore one **orchestrator-owned**
  stage. Before either process starts, the shell/canonical harness invokes the
  package-safe `sage3d.cli.create_staging` wrapper around the same helper,
  captures its returned path, and passes that exact existing directory to RGB,
  depth, and the finalizer. Individual module-render processes never allocate or
  accept an absent staging root. Phase 0b legacy capture predates this production
  CLI: its checker-only harness uses an equivalent isolated stdlib sibling
  allocator, which is retired in favor of the production helper/CLI when they
  land in Phases 1 and 4.

Cross-filesystem copy fallback is forbidden because it would weaken atomicity;
fail with an actionable error instead. Concretely, package staging for an
`/ssd5/.../<scene>` target also lives under that `/ssd5` parent, never under the
shell's `/tmp` work root.

**Filesystem-entry safety is binding:** “target absent” means
`os.path.lexists(target) == False`; existing files, directories, symlinks, and
dangling symlinks are all refusal states. Internally allocated generate/package
stages and the orchestrator-allocated render stage use
`create_staging_directory`; staging must be a real directory, never a symlink.
Resolve the staging/target parents strictly and require the same intended real
parent/device. Before validation/publication, walk staging with `lstat` and
reject symlinked modality directories or artifact files plus FIFOs, sockets,
devices, or other unexpected entry types. A configured parent path may traverse
a symlink only when its strict resolution is the validated intended parent;
target and staging entries themselves may not.

Publication tests cover existing empty/non-empty directories, regular files,
symlinks, dangling target symlinks, staging-root symlinks, symlinked render
images/summaries, unexpected special entries, device mismatch, and a target
injected immediately before the final absence recheck/rename. The documented
race after that recheck remains outside the cooperative-publisher contract.

Publication supports **one cooperative publisher per target**. It rechecks
target absence immediately before `os.rename` and tests both empty and non-empty
existing targets, but does not claim race-safe no-clobber behavior against a
non-cooperating process that creates the target between check and rename.
Concurrent publication to the same target is explicitly outside the contract;
do not add implicit replacement, copy fallback, or a portability claim for
`renameat2(RENAME_NOREPLACE)`.

Atomic publication guarantees that no partial final tree is visible under the
cooperative single-publisher/process-failure model. It does **not** claim
persistence across sudden power loss unless file and parent-directory `fsync`
durability is separately implemented and tested; full power-loss durability is
not required by this refactor.

- **Generate (`write_trajectory_artifacts`):** internally allocate a sibling
  stage → validate → atomic rename into the absent target.
- **Package (`package`):** validate the source pipeline contract → stage the
  complete dataset in an internally allocated sibling directory → run
  `validate_packaged_dataset(staging_root, trajectory_dir, rendered_dir,
  config)` → atomic rename only after validation succeeds. The production
  validator checks inventory, Arrow/JSON content, copied-input checksums,
  calibration/extrinsics, and depth metadata without invoking CLI/publication
  behavior. **The shell no longer pre-creates the final output dir** and may run
  the external checker after publication as independent confirmation.
- **Render (two processes, one staged root):** RGB and depth both write into one
  orchestrator-allocated sibling staging directory, e.g.
  `.rendered.<allocator-id>`; they never write into the final `rendered/`
  target. Each modality requires the same existing real directory, verified
  with `lstat`; it never creates that root. Its own image directory and summary
  files must be absent. Before writing, a strict preflight accepts only an empty
  root or the exact, fully valid inventory of the other modality; partial,
  unrelated, or same-modality entries are rejected. After both processes
  finish,
  `cli/finalize_render.py` runs the complete render/trajectory contract against
  the staging root and atomically renames the **entire directory** to the absent
  `rendered/` target. The staging and target paths must be siblings on the same
  filesystem.
- **Accepted module-render stage states:** a mode may start from an existing
  empty root, complete RGB-only inventory, or complete depth-only inventory. It
  may add only the missing modality. An absent root is rejected. Finalization
  accepts only the exact complete two-modality inventory. Any partial modality,
  duplicate own modality, unrelated path, or summary/image mismatch is rejected.
- **Failure/restart semantics:** an exception or process death may leave an
  incomplete staging directory, but the final target remains absent. Producers
  never clean or reuse ambiguous partial staging state. The shell allocates a
  new staging path for every attempt (and may remove an old one only under its
  explicit replacement ownership); direct callers must inspect/remove the
  partial stage or allocate a new one. Canonical failed-run staging is retained
  under the evidence retention policy long enough for diagnosis. Tests cover
  exceptions, process-restart fixtures with pre-seeded partial states, refusal
  to overwrite a modality within staging, and final-target non-overwrite.

**Destructive-shell guardrails:** `SceneConfig` and the shell accept scene IDs
matching `^[0-9]+$` only. `OUTPUT_ROOT` is durable/operator-owned and must
already be a nonempty path string naming an existing real directory by `lstat`
(the directory itself may be empty and must not be a symlink).
`WORK_ROOT` is disposable and may be created safely when absent: its path and
basename must be nonempty; the basename must be one component other than `.`
or `..`; its immediate parent must already exist, pass `lstat` as a real
directory, and resolve with `realpath -e`; `os.path.lexists(WORK_ROOT)` must be
false; create exactly that one directory exclusively (plain `mkdir`, never
`mkdir -p`); then verify it with `lstat` and resolve it. If `WORK_ROOT` already
exists, it must pass the same real-directory checks. This preserves the default
`/tmp/opencode/sage3d_pointgoal` behavior while typo-protecting durable output.

Before any `rm -rf`/replacement, resolve the existing target parent with
`realpath -e`; if the target exists, reject a symlink by `lstat` and resolve it
with `realpath -e`, otherwise form the candidate from the resolved parent plus
the validated single-component scene name. Require the result to be a strict
descendant of its declared root. Refuse empty paths, `/`, the configured root
itself, `..` traversal, symlinked roots/targets, or any unexpected resolved
parent. To protect source, refuse every destructive target equal to **or nested
under** the resolved `SCRIPT_DIR`. Print the exact validated target immediately
before deletion. Tests cover safe creation of the absent default/custom work
root, a missing/invalid work-root parent, pre-provisioned output-root
requirements, empty/root/self paths, source-tree descendants, `..`, nonnumeric
scenes, symlinked roots/targets, and valid descendants. The Phase 0b canonical
harness uses isolated validated work/evidence roots and never delegates
destructive cleanup to the current unguarded legacy `--force` path.

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
                           #   create_staging_directory + same-filesystem
                           #   atomic directory publication
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
                           #   encoded_depth_sentinel, encode_depth,
                           #   RawDepthSummaryAccumulator
  render_bootstrap.py      # [Isaac] pure imports; parse config, start SimulationApp,
                           #   import render_runtime, close in finally
  render_runtime.py        # [Isaac] imported AFTER app: RenderMode strategy,
                           #   render_episode, render(config, staging_root)
  lerobot_dataset.py       # [pyarrow] build_episode_parquet, copy_episode_frames,
                           #   write_lerobot_meta, validate_packaged_dataset,
                           #   package (stage→validate→rename)
  cli/
    __init__.py            # package marker (explicit; no namespace-package reliance)
    _args.py               # add_fisheye_args, add_scene_args
    generate.py            # Phase 3e
    render.py              # Phase 4
    create_staging.py      # Phase 4; package-safe render-stage allocator
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
tests/golden/<scene>/<baseline-id>/  # immutable Phase 0b policy/provenance/goldens
scripts/check_{generate,package,render}.py   # Phase 0b (checker scripts; not in the
                                             #   module-creation audit, which covers
                                             #   production modules + config only)
scripts/run_sage3d_canonical.py              # Phase 0b canonical harness; writes
                                             #   run provenance, checker results,
                                             #   and final verification manifest
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
is in scope. Executable `--help` smoke tests for `sage3d.cli.generate`,
`create_staging`, `render`, `finalize_render`, and `package` land in Phases
3e / 4 / 4 / 4 / 5 respectively; Phase 2c imports only then-existing modules
from outside the repo under both interpreters.

### Render/finalizer CLI and compatibility contract (decided)

The new pipeline uses this exact boundary (camera arguments repeated for both
render processes; shown explicitly to avoid implicit shared state):

```bash
RENDER_STAGE="$(
  PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$SAGE3D_PACKAGE_PYTHON" -m sage3d.cli.create_staging \
    --final-target "$RENDERED_DIR" --prefix ".rendered."
)"

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

`sage3d.cli.create_staging` prints exactly one absolute allocated path to stdout
(diagnostics go to stderr), refuses an existing final target, and delegates all
allocation/lstat/parent/device checks to
`publication.create_staging_directory`. `sage3d.cli.render` accepts
`--staging-root`, requires it to be an existing real directory, and does
**not** accept `--output-dir`; the allocator and finalizer enforce its sibling
relationship to the final target, and `finalize_render` alone accepts both
staging and final output paths.

The legacy `render_fisheye_sage3d.py` shim retains its old `--output-dir`
surface by mapping that exact path to the new renderer's staging root, emits an
actionable deprecation/non-atomic warning, and never invokes the finalizer. As
an explicit compatibility exception, if the legacy output path is absent the
shim may create that exact directory with exclusive `os.mkdir` (not `mkdtemp`
or `mkdir -p`) and immediately verify it with `lstat`; a later legacy modality
may consume the existing valid other-modality state. Thus two direct legacy
invocations preserve the legacy artifact inventory but are an explicit
non-atomic compatibility exception; only the allocator + module-CLI +
finalizer sequence claims validated atomic publication. Automatic “second mode
finalizes” behavior is forbidden.

### Golden storage, tolerance capture, and re-baselining

Before formal handoff, commit this revision 8 plan and record its Git commit plus
the **baseline source commit** in Phase 0b provenance. In git: provenance
manifest, input-asset SHA-256 values, exact baseline commands, normalized
semantic config, expected inventory, deterministic selected-frame list, small
RGB/depth golden frames, per-run metrics, threshold-derivation report, and all
pinned tolerances. Also record
OS/kernel; both Python versions; Isaac Sim/Kit/USD; GPU model, driver,
CUDA/runtime and renderer settings; and exact NumPy, SciPy, trimesh, rtree,
OpenCV, Pillow/libjpeg, and PyArrow versions in the environment where each
stage runs. Outside git: the full artifacts, referenced by immutable URI/cache
key and checksum.

**Recorded identity/audit fields — not equality-required:** baseline source
commit, candidate source commit, baseline and candidate exact commands/entry
points, worktree state, and submodule state. Candidate code and entry points are
expected to differ during this refactor. **Binding canonical evidence requires
both baseline and candidate to be clean, Git-resolvable commits.** Dirty runs
may execute every test but are report-only. Generated artifacts/evidence live
outside the source tree or under explicitly ignored paths. A source-commit or
literal-command difference never makes golden thresholds report-only by itself.

**Equality-required common fields:** input-asset hashes; canonical trajectory
identity for render/package comparisons; normalized semantic stage config;
selected-frame policy/list; and the applicable stage-specific runtime
fingerprint below. Normalized render config includes scene/asset identities,
resolution, camera model/fisheye calibration, min/max depth, depth scale,
startup/settle steps, render mode, and trajectory identity. Generation/package
comparisons analogously include every behavior-affecting semantic field, not
literal command spelling. Package comparison binds trajectory identity,
pointcloud identity, and stable render config/summary authority; candidate
nondeterministic rendered-root hashes are recorded for package↔current-input
copy checks but are **not** required to equal baseline media hashes.

| Stage | Equality-required runtime fingerprint |
| --- | --- |
| Generate | Python build/version; CPU architecture; NumPy; OpenCV; SciPy; trimesh; rtree; Pillow; pxr/USD; numerical backend identity; `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` including explicit `<unset>` values |
| Render | Python build/version; NumPy; Pillow/libjpeg; Isaac Sim/Kit/USD; GPU model; driver; CUDA/runtime; renderer build/settings; allowlisted renderer-affecting environment variables |
| Package | Python build/version; NumPy; PyArrow; Pillow when image decoding participates in validation |
| Golden checker | Python build/version; NumPy; Pillow/libjpeg and PyArrow as used by the invoked checker |

Phase 0 records a human-readable environment export and a digest for each
interpreter alongside these fields; an opaque environment digest does not
replace the diagnostic versions above. OS/kernel and other allowlisted
environment/backend fields are recorded diagnostically and promoted to
equality-required when Phase 0 sensitivity testing demonstrates output impact.
Trajectory identity uses a canonical content digest over ordered episode/array
keys, dtypes, shapes, C-order bytes, and parsed manifest semantics—not NPZ
container metadata, paths, or timestamps.
Normalized config is schema-versioned canonical JSON with defaults resolved,
stable field/list ordering, explicit numeric/boolean types, and paths represented
for equality by their declared asset/trajectory identities rather than host-local
spellings; original paths remain audit-only fields.

**Canonical digest framing (binding):** every canonical artifact/tree digest is
SHA-256 over a domain-separated, schema-versioned byte stream; do not hash
ambiguous concatenated strings. Text is UTF-8. Every variable-length name,
metadata block, and payload is preceded by an unsigned 64-bit big-endian byte
length, and collections begin with an unsigned 64-bit big-endian item count.
Directory components are ordered by normalized relative POSIX path after
rejecting absolute/parent-traversal paths, duplicate normalized names,
symlinks, and non-regular/non-directory entries.

Canonical JSON bytes are exactly
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False).encode("utf-8")`; JSON object order is semantic-insensitive,
while list/episode order remains significant. NaN/Infinity are rejected rather
than emitted as non-standard JSON. Each array component frames its normalized
episode/file name, NPZ key, `numpy.dtype.str` (including byte order), rank and
shape dimensions, the literal storage order `C`, payload length, and contiguous
C-order bytes. The top-level domain tag is `sage3d-digest-v1` plus the digest
kind (trajectory, rendered root, packaged root, or evidence); cross-kind reuse
is invalid. `pointcloud.ply` identity is its exact file SHA-256—no semantic PLY
alternative—because generation and package-copy contracts preserve exact
bytes. Phase 0b commits independently reproducible positive/negative digest
test vectors and their expected hex digests in provenance.

**Actual-run provenance:** `scripts/run_sage3d_canonical.py --evidence-dir ...`
automatically writes `<evidence-dir>/run_provenance.json` outside the
behavior-preserved artifact roots, never by reconstructing missing fields from
summaries:

```json
{
  "schema_version": 1,
  "plan_revision": 8,
  "plan_commit": "...",
  "baseline_id": "...",
  "candidate_commit": "...",
  "dirty_tree": false,
  "submodule_state": {},
  "normalized_config": {},
  "input_hashes": {},
  "artifact_digests": {},
  "runtime_fingerprint": {},
  "cache_policy": {},
  "stage_runs": [
    {
      "stage": "render-rgb",
      "pid": 12345,
      "argv": ["python", "-m", "sage3d.cli.render"],
      "cwd": "...",
      "environment": {},
      "started_at": "...",
      "completed_at": "...",
      "exit_code": 0
    }
  ]
}
```

`compare-golden` consumes this as `--run-provenance`; it compares only the
equality-required fields with `--baseline-provenance`, while retaining the
identity/audit fields in its report. The canonical harness computes hashes and
runtime fields automatically and captures only an explicit environment
allowlist—never secrets or unrelated variables. Before comparison it verifies:
(1) one `baseline_id` across baseline provenance, tolerance policy, baseline
artifact manifest, and selected-frame directory; (2) baseline directory
inventory/checksums; (3) candidate roots against `artifact_digests`; (4) the
actual supplied trajectory root against the recorded trajectory digest; (5) the
actual rendered root used by package validation against its recorded identity;
and (6) successful `exit_code == 0` completion records for every required
producer subprocess. Manually supplied fingerprint claims are not authoritative.

Write the final `run_provenance.json` atomically after all required producer
stages succeed and before invoking golden checkers. Failed-run diagnostics use a
distinct non-binding partial/status file.
On an equality-required or evidence-chain mismatch, structural/logic and
synthetic tests remain binding, but golden thresholds are report-only and cannot
replace canonical PR evidence.

After every required checker finishes, the canonical harness atomically writes
`<evidence-dir>/verification_manifest.json` as the final binding evidence unit:

```json
{
  "schema_version": 1,
  "baseline_id": "...",
  "candidate_commit": "...",
  "run_provenance_sha256": "...",
  "baseline_provenance_sha256": "...",
  "tolerance_policy_sha256": "...",
  "results": [
    {
      "checker": "check_generate",
      "mode": "compare-golden",
      "result_sha256": "...",
      "exit_code": 0,
      "eligible": true,
      "artifact_digests": {}
    }
  ],
  "overall_eligible": true
}
```

Every checker emits an atomic JSON result that identifies the exact artifact
root digests it evaluated. The verification manifest checksum-binds all required
checker results, run provenance, baseline provenance, and tolerance policy;
`overall_eligible=true` is permitted only when every required command completed
successfully and returned `eligible=true`. Failed/incomplete orchestration
writes a separate non-binding status file and never a successful verification
manifest. Tamper tests cover substituted/stale checker results, artifacts,
policies, provenance, commits, and baseline IDs. `run_provenance.json` remains
immutable producer evidence; the final manifest does not rewrite it.

**Tolerance-capture protocol (binding before Phase 0b):** allocate the immutable
baseline ID and commit
`tests/golden/839920/<baseline-id>/tolerance_policy.json` before inspecting
metric results. It pins the metric formulas above,
`rgb_mask_dilation_pixels`, separate pre-observation minimum margins
(`rgb_leakage_min_margin`, `rgb_rmse_min_margin`,
`rgb_p99_min_margin`, `depth_error_min_margin`, `depth_iou_min_margin`), and
these deterministic frames: first/middle/last frame (integer floor for middle)
from **every pinned episode**, de-duplicated within an episode while preserving
order. For each metric it also declares: the smallest intended detectable
regression, benign GPU variation that must pass, a boundary mutation, and
whether the threshold is evaluated per frame, per run, or globally. For five
episodes this is fifteen RGB/depth pairs.

Every designated-baseline, characterization, and held-out render pair uses
fresh RGB/depth processes, a newly allocated shared staging root, no reused app
state or output files, identical semantic inputs, and recorded process IDs plus
start/end timestamps. Record the filesystem/shader cache policy and, when
practical, include at least one documented cold-cache or machine-restart
characterization; otherwise record that limitation explicitly. Failed-run
stages and partial evidence are retained for a documented diagnostic period
rather than automatically deleted. Then:

1. Capture one designated golden generate/render/package baseline.
2. From the same trajectories, run **five** additional fresh RGB/depth
   render pairs on the matching canonical setup and compute every selected-frame
   metric against the designated baseline.
3. For each upper-bound metric, pin
   `max_observed + max(0.25 * max_observed, <metric>_min_margin)` using its own
   named margin above. For depth IoU, pin
   `max(0, min_observed - max(0.25 * (1 - min_observed),
   depth_iou_min_margin))`.
   Leakage uses the worst absolute observation across baseline + characterization
   runs; difference metrics use characterization-versus-baseline observations.
4. Run **two** additional fresh held-out RGB/depth pairs. Neither contributes to
   threshold derivation and both must pass every derived threshold. A failure
   invalidates the capture attempt, reopens investigation, and requires a fresh
   complete protocol—not appending the failed held-out run to characterization.
5. When full baseline/candidate renders are available, calculate report-only
   all-frame metric distributions in addition to the binding selected frames.
6. Store per-frame/per-run values, formulas, margins, thresholds, and held-out
   results; the engineering owner approves the report at the Phase 0b exit gate.

**Golden-checker mutation suite:** before accepting thresholds, apply fixed
mutations to selected frames and record the structural check/metric expected to
detect each one. Must-fail RGB cases: all-black, near-uniform, channel swap,
horizontal/vertical flip, one-frame offset, removed/changed forward mask,
localized block corruption covering more than 1% of evaluated samples, and a
multi-code global intensity shift. Must-fail depth cases: all sentinel, wrong
scale, missing sentinel behavior, horizontal/vertical flip, one-frame offset,
changed outside-mask value, eroded/expanded non-max mask, and constant encoded
offset. A global one-code RGB shift is a documented boundary probe whose
expected pass/fail result follows the committed margin policy; do not require it
to fail if it is intentionally within tolerance. Every unexpected mutation pass
invalidates threshold acceptance and triggers investigation.

Mutation fixtures must make rejection deterministic. Establish and assert a
minimum channel-difference precondition before channel swapping; choose
one-frame-latency probes with sufficient baseline frame-to-frame metric
separation and use a precommitted larger offset fallback; and pin exact
corruption sizes/magnitudes for intensity shifts, depth offsets, mask
erosion/expansion, and block corruption. If a realistic one-frame latency is
below the pinned scene's tolerant metric threshold, record that limitation and
rely on the exact mocked call-trace test for ordering rather than loosening or
tuning the golden threshold.

Boundary-budget fixtures additionally include: narrow JPEG-block-edge RGB mask
leakage expected to pass; leakage beyond the committed dilation radius expected
to fail; widespread one-code and localized high-intensity exterior leakage with
predeclared outcomes; a small depth-mask boundary perturbation expected to pass;
and meaningful depth-mask-area loss expected to fail. The tolerance report maps
each case to the metric and evaluation scope that permits or rejects it.

**Controlled re-baselining:** never edit a baseline in place. A changed source
commit or entry point alone does **not** trigger re-baselining. For an approved
equality-required input/config/runtime change, use an explicit provenance-update
PR that retains the old baseline/metrics, explains every binding change, runs
old and new environments where available, repeats the complete capture
protocol, obtains independent approval under the policy above (or records the
explicit exception), and assigns a new immutable
baseline/provenance ID.

### Checker execution contract (two explicit modes)

All `scripts/check_*.py` programs are read-only, run under package Python, emit
a concise human/JSON result, and exit nonzero on binding failure.

- **`validate` mode** is baseline-independent and valid for arbitrary supported
  shell inputs. `check_generate.py` takes `--trajectory-dir`;
  `check_render.py` takes `--rendered-dir --trajectory-dir`; and
  `check_package.py` takes
  `--dataset-dir --trajectory-dir --rendered-dir`.
- **`compare-golden` mode** first runs `validate`, then also requires
  `--baseline-dir --baseline-provenance --run-provenance`, verifies only the
  equality-required inputs/config/runtime fields, and applies the canonical
  deterministic/tolerant assertions. It refuses golden comparison when those
  semantic fields differ; candidate source commit and literal entry-point
  differences are reported but expected.
  A dirty or non-resolvable baseline/candidate commit may still produce metric
  diagnostics, but the result is explicitly `eligible=false` and cannot satisfy
  a canonical gate.

`check_generate.py validate` proves schema/inventory plus manifest↔NPZ↔PLY/viz
structural and cross-artifact consistency; it cannot prove exact generated
arrays without a reference. Its golden mode applies the complete generation
extraction oracle, including exact arrays and deterministic artifact comparison.
`check_render.py validate` proves inventory, calibration, trajectory linkage,
encoded-depth structure, and summary shapes/ranges/counts; it does not
reconstruct raw-depth aggregate formulas from clipped PNGs. Synthetic
accumulator tests and render-runtime capture/call tests prove those formulas;
render golden mode adds the named tolerant RGB/depth regression metrics.
`check_package.py validate` implements the baseline-independent package oracle
and reuses production low-level validation primitives where practical without
importing a CLI or publication behavior; its golden mode adds only deterministic
package comparisons and excludes render-derived baseline bytes.

Canonical orchestration runs `check_render.py compare-golden` and
`check_package.py compare-golden` as independent required commands. The package
checker neither invokes the render checker nor consumes a separate render-result
file; copied render artifacts are checked only against the current rendered
root. The normal shell always runs `check_package.py validate` with the current
three roots. After rewiring, canonical phase DoDs require fresh artifacts and
the applicable independent `compare-golden` commands; merely rechecking the old
snapshot is insufficient.

Because Phase 0b precedes the target package, its checkers initially own
standalone read-only helpers and import no not-yet-created `sage3d` module. In
Phase 5, move/extract the common packaged-artifact primitives into
`lerobot_dataset.py`, rewire `check_package.py validate` to them, and use the
Phase 0b oracle to prove checker behavior did not change. The checker must not
import `cli.package` or call publication code.

The depth sentinel follows the same staged extraction: Phase 0b
`check_render.py` temporarily owns one standalone legacy helper plus the pinned
numeric regression vectors. Phase 1 moves the canonical implementation to
`sage3d.render_processing.encoded_depth_sentinel`, rewires the checker to import
it, and proves identical results with those vectors. From Phase 1 onward, an
independent checker sentinel formula is forbidden.

**Canonical-digest acceptance tests (Phase 0b):** the trajectory digest is
invariant to NPZ compression/container bytes, mtimes, artifact-root relocation,
JSON whitespace, and host-path spelling changes in exactly `scene_dir` and
`collision_usd`. It changes for any array value/shape/dtype/byte-order change;
episode add/remove/rename/reorder; NPZ key change; non-path manifest-field
change; or manifest episode-order change. Normalize exactly the two named
manifest path fields; never heuristically remove other “path-like” strings.
Package input identity additionally includes the exact
`pointcloud.ply` file SHA-256; visualization files are excluded from
render/package eligibility. Tests cover every invariance/sensitivity case,
framing boundary, component/path ordering, canonical-JSON rejection, and the
committed digest vectors explicitly.

---

## Implementation phases

Each phase is independently mergeable. Tests for a new API **co-land with the
phase that creates it**; DoDs import only modules that exist at that phase.

**Architecture freeze after revision 8:** implement Phases 0–6 through
dependency-ordered PR/ticket checklists derived from these DoDs. Phase 0 factual
discoveries update the applicable ticket and immutable decision/provenance
record. Reopen this architecture only for contradictory new evidence or a
scope/version change approved by the owner; do not add further broad narrative
design during routine implementation.

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
7. Named RGB/depth metrics, deterministic selected frames, threshold formulas,
   and the repeated-run/held-out protocol above are committed before Phase 0b;
   numeric tolerances are established only through that protocol.
8. **Handoff identity:** revision 8 is committed before baseline capture; the
   baseline provenance identifies that plan/source commit, while each candidate
   sidecar separately identifies the code it actually executed.
9. **Approval ownership:** the Phase 0 decision record names accountable owners
   for baseline/tolerance approval, canonical GPU evidence, re-baselining,
   package-format sign-off, and compatibility-shim removal sign-off. At least one
   approver other than the implementation author is required for the
   pre-observation tolerance policy, Phase 0b threshold report, re-baselining,
   Phase 4 render extraction, and package-format compatibility sign-off. Other
   roles may share a person, but none may be left implicit. When staffing makes
   independent review impossible, record an explicit owner-approved exception
   with scope and rationale; one-person approval is not the default policy.

### Phase 0a — Legacy characterization tests (no target modules)  *(~1 day)*

Per the decisions above. Commit and obtain owner approval for
`tests/golden/839920/<baseline-id>/tolerance_policy.json` (metric
names/formulas, mask construction, deterministic frame selector, threshold
formulas, per-metric minimum margins, five-characterization/two-held-out run
count, per-metric detection budgets/evaluation scopes, cache/isolation policy,
diagnostic retention, and deterministic mutation expectations/preconditions)
before any Phase 0b render metrics are inspected. **DoD:** green on the
unmodified codebase
(no-asset tests) under package python; tolerance policy reviewed and immutable
for the first capture attempt; all five approval owners recorded; and an
independent reviewer approves the policy or the explicit exception is recorded.
**Risk:** low.

### Phase 0b — Pinned external baselines + artifact checkers  *(~3–5 days)*

First run a generation-only feasibility smoke using scene `839920`, seed
`20260720`, five episodes, and the current generation defaults; stop before the
expensive render baseline if it cannot complete within `max_attempts=3000`.
Then implement both modes of `check_{generate,package,render}.py`; capture the
designated pinned generate/render/package baseline; run five characterization
and two held-out RGB/depth pairs; derive the named tolerances with the committed
formulas; execute the mutation suite; and package/validate held-out outputs
against their current inputs. Run a fresh deterministic generation comparison
as well. Create
`scripts/run_sage3d_canonical.py`; it writes baseline/candidate
`run_provenance.json` sidecars with normalized config and actual
hashes/fingerprints, captures atomic checker JSON results, and writes the final
`verification_manifest.json`; checker tests prove that a
different candidate commit/entry point remains golden-eligible while a changed
equality-required field is rejected. Phase 0b `check_render.py` owns only the
temporary standalone sentinel helper and pinned numeric vectors described above.
The Phase 0b canonical harness likewise owns only a temporary stdlib render-stage
allocator with the same sibling/lstat/device contract and saved filesystem test
vectors; it does not import not-yet-created production modules.
Add evidence-chain tamper tests for every baseline/candidate ID, digest, root,
subprocess-status, checker-result, and final-manifest check; add the complete
digest framing/invariance/sensitivity matrix and committed test vectors; and
verify canonical work/evidence roots satisfy the destructive-path guards.
Changing scene/seed/count requires an explicit plan/provenance update, not an
implementer-local substitution. **Formal exit gate before Phase 1:** all three
checker `validate` modes pass; canonical fresh artifacts pass
`compare-golden`; both held-outs and the mutation-sensitivity report pass and
are approved; required provenance fields are complete; the canonical GPU
fingerprint is captured; baseline/candidate commits are clean and resolvable;
all stage-specific fingerprints and structured successful subprocess records
are present; every characterization/held-out run satisfies the process,
staging, timestamp, cache-policy, and retention requirements; an atomic
`verification_manifest.json` binds all eligible checker evidence; the
threshold report has independent approval (or a recorded exception); and the
owner accepts a revised schedule/risk estimate based on
measured runtime, retry rate, artifact size, and checker complexity. **Risk:**
medium until the gate closes.

### Phase 1 — Package-safe leaf modules + wiring  *(~1.5 days)*

**Create:** `__init__.py`, `frames.py`, `camera.py` (+ width/height/FOV and
finite-value validation), `episode_arrays.py`, `naming.py`, `io_ply.py` (writer + parser),
`pointcloud.py` (`voxel_downsample`), `publication.py`
(`create_staging_directory`, absent-target, filesystem-entry, and
same-filesystem atomic-directory helpers), `render_processing.py`
(`build_forward_mask`, `mask_rgb`, `encoded_depth_sentinel`, `encode_depth`,
`RawDepthSummaryAccumulator` with the contracts above), `cli/__init__.py`,
`cli/_args.py` (`add_fisheye_args`).
**Wiring (behavior-preserving; CLI surfaces unchanged):** render uses
`frames`+`CameraCalibration`+`naming`+`render_processing` (preserve Isaac
readback), moves the Phase 0b sentinel helper into `render_processing`, rewires
`check_render.py` to import it, and preflights it for both render modes before
`SimulationApp` or any output artifact write; package uses
`CameraCalibration`+`naming`; generate writes npz via
`EpisodeArrays`, uses `frames.yaw_to_rotation2d`+`pointcloud`+`io_ply`+`naming`.
**DoD:** Phase 0a/0b checks pass; **forbidden-import smoke test (Phase 1 module
list only)** green in fresh subprocess; checker no longer contains an
independent sentinel formula; saved divergent/overflow vectors match before and
after migration; RGB and depth invalid-config tests fail before app launch or
render artifact writes; both checker modes green; unit tests for
frames/camera/naming/io_ply/pointcloud/publication/render_processing green,
including the binding `lexists`/symlink/special-entry publication matrix.
Migrate the Phase 0b harness's allocator to
`publication.create_staging_directory` and prove the saved filesystem vectors
unchanged; from Phase 1 onward, an independent canonical-harness allocation
formula is forbidden.
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
adds `config`,`schemas`; render `validate` and canonical `compare-golden` green.
**Risk:** medium.

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
3e/4/5); render `validate` and canonical `compare-golden` green. **Risk:** low.

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
file/symlink/dangling-symlink/special-entry refusal plus
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
finite positive depth scale, `0 < min < max`; **package-safe** stdlib-only basic
validation, with float32 overflow authority in `encoded_depth_sentinel`);
`render_bootstrap.py` (pure imports → parse config → require/validate the
existing orchestrator-owned staging root → sentinel preflight for
either mode before any staging write → construct
`SimulationApp` → import `render_runtime` → run → **close in `finally`** incl
failures during runtime imports/stage setup); `render_runtime.py` (`RenderMode`
strategy: `.build_stage`/`.configure_camera`/`.capture`, single `render_episode`
loop using `render_processing`, `render(config, staging_root)`); `cli/render.py`;
package-safe `cli/create_staging.py` (thin stdout-only-path wrapper around
`publication.create_staging_directory`);
`cli/finalize_render.py` (load staged artifacts → run the full contract → call
`publication.atomic_publish_directory`). Keep the **two-process** model (a mode
does not "spawn" an app). `render_fisheye_sage3d.py` → rollout shim. The shell
first invokes the allocator under `SAGE3D_PACKAGE_PYTHON`, then runs RGB and
depth against the returned sibling staging root, and finally invokes the exact
finalizer command under `SAGE3D_PACKAGE_PYTHON`; new CLI and legacy-shim
behavior follow the binding contract above. **DoD:** render oracle green (all
10 items incl exact sentinel/`encode_depth`, selected golden depth, streaming
accumulator, mocked stage construction with
**exact reference string incl `[gauss.usda]`**, the complete warmup/capture call
trace, pre-encode mask, named/directional RGB metrics, and app-close-on-failure);
RGB/depth invalid sentinel inputs fail before app construction and leave staging
untouched;
failure injection before/within each modality and immediately before final
publication leaves the final target absent; pre-seeded partial-stage and
same-modality-overwrite tests green; finalizer rejects incomplete/stale/invalid
inventories, symlinked staged entries, and every existing-target entry type;
module render rejects an absent or symlink staging root, allocator output is an
existing real sibling directory, and legacy exclusive exact-path creation is
covered separately; the canonical harness switches from direct helper use to
the production CLI and captures its path without parsing diagnostic output;
`python -m sage3d.cli.create_staging --help`,
`python -m sage3d.cli.render --help`,
and `python -m sage3d.cli.finalize_render --help` work outside the repo; an
independent reviewer approves the render extraction or the explicit exception
is recorded.
**Risk:** medium–high until logic/sequencing/stage/publication tests land.

### Phase 5 — Package extraction + explicit format contract  *(~2 days)*

**Create:** `config.PackageConfig` (positive FPS, path/output requirements,
optional compatibility-assertion fields); `lerobot_dataset.py`
(`build_episode_parquet`, `copy_episode_frames`, `write_lerobot_meta`,
`validate_packaged_dataset`, `package` with sibling
stage→validate→atomic-rename; non-destructive); `cli/package.py`; a
checked-in format note documenting the project-specific LeRobot-style layout
and the separate standardization follow-up. Package consumes only the finalized
render root, constructs calibration and depth metadata from the canonical depth
summary, and preserves the default baseline output exactly. Add the
non-default-depth-scale test plus the complete float-policy positive/negative
matrix. Move the common Phase 0b package-checker primitives into
`lerobot_dataset.py` and rewire the checker as specified above. The shell no
longer pre-creates final output;
`--force` removes it before invoking package and switches to
`python -m sage3d.cli.package`. `package_lerobot_sage3d.py` → rollout shim.
If Phase 0a found a real downstream consumer, its pinned smoke test is required;
otherwise the semantic package oracle is binding and the lack of a named
consumer is recorded, not left as an implementer choice. Run
`prepare_trajectories.py` as a supplemental smoke for shared Parquet path,
`observation.camera_extrinsic`, and `action` only; failure is actionable, but it
does not replace the SAGE3D package oracle. **DoD:** package oracle
green **including staged and post-publication
extrinsic/calibration/depth-metadata checks**; staging validation must fail
before publication under each negative/failure-injection fixture;
`check_package.py validate` passes on
arbitrary synthetic/noncanonical configs and both modes pass on canonical
fresh artifacts; exact inventory and default metadata preserved;
non-default depth scale truthful; failure-injection, partial-staging, rerun, and
the complete filesystem-entry/existing-target refusal matrix green;
`python -m sage3d.cli.package --help`
works outside the repo; package-format compatibility receives independent
sign-off or the explicit exception is recorded. **Risk:** low–medium.

### Phase 6 — Integration, portability, documentation, rollout  *(~1 day)*

Finalize `run_pipeline_sage3d.sh`: derive `SCRIPT_DIR`; honor
`SAGE3D_ISAAC_PYTHON`/`SAGE3D_PACKAGE_PYTHON` with documented local defaults;
prepend the binding `PYTHONPATH`; allow only the guarded single-component
creation of disposable `WORK_ROOT` while requiring operator-provisioned
`OUTPUT_ROOT`; allocate the shared render stage with the package-safe allocator
beside `RENDERED_DIR`; run RGB → depth → package-Python finalizer → package;
let generate/package allocate their own sibling stages; never pre-create
publication targets; and confine destructive cleanup to explicit
shell-owned `--force`/work-directory operations protected by the binding
numeric-scene/canonical-root/strict-descendant/symlink guardrails above.
Preserve `--plan-only` as the
generate-only early exit. Replace the current inline frame/parquet inventory
validator with the baseline-independent command:

```bash
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$SAGE3D_PACKAGE_PYTHON" "${SCRIPT_DIR}/scripts/check_package.py" validate \
  --dataset-dir "$SCENE_OUTPUT" --trajectory-dir "$TRAJECTORY_DIR" \
  --rendered-dir "$RENDERED_DIR"
```

Its concise success summary replaces the old counts; do not maintain duplicate
validation logic or invoke canonical comparison for arbitrary shell inputs.
Update README with
module CLI examples, environment setup, project-specific package compatibility,
staging/recovery rules, and output authority. Run the pinned full pipeline from
a CWD outside the repo; the canonical harness independently runs generation,
render, and package `compare-golden` with the same candidate run-provenance
identity, archives all validation evidence, and writes the final atomic
verification manifest. **DoD:** all three test lanes and all checker modes
green; absent-default-work-root, pre-provisioned-output-root, source-subtree
refusal, and allocator-to-both-render-process integration tests pass; legacy
shell and shim invocations preserve
exit codes/default artifacts apart from documented safer existing-output
refusal and deprecation notices; malicious/accidental deletion-path tests green;
no hardcoded repository path remains.

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
| `publication.py` | 1 | pkg | stdlib+pathlib; staging allocator + absent-target + atomic directory publish |
| `render_processing.py` | 1 | pkg | owns float32 `encoded_depth_sentinel`, `encode_depth`, and streaming `RawDepthSummaryAccumulator`; consumed by `render_runtime` in 4 |
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
| `render_bootstrap.py`, `render_runtime.py`, `cli/render.py` | 4 | Isaac | bootstrap/runtime split; require orchestrator-owned stage |
| `cli/create_staging.py` | 4 | pkg | allocate/print one shared render stage |
| `cli/finalize_render.py` | 4 | pkg | validate complete staging root, then publish |
| `PackageConfig`, `lerobot_dataset.py`, `cli/package.py` | 5 | pkg | project-specific format contract; includes staged-dataset validator |
| Shell/docs rollout | 6 | mixed | portability, recovery, deprecation, full integration |
| Phase 7 perf items | 7 | varies | one PR each, with acceptance criteria |
| shim removal | post-rollout | — | separate PR after the Phase 6 removal gate |

> Scope: this table covers **production package modules + config classes only**.
> Phase 0b checker/canonical-harness scripts and `tests/` helpers are not
> production modules.

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

**Large-phase PR decomposition (binding review guidance):** a phase is a gate,
not necessarily one PR. Phase 1 splits into (a) package-safe leaf/publication
utilities, (b) render processing + sentinel migration, (c) generation wiring,
and (d) render/package shared camera/naming wiring. Phase 4 splits into
(a) bootstrap/app lifecycle + mocked adapters, (b) stage strategies/unified
episode loop, (c) render staging/preflight, (d) finalizer/shell integration,
and (e) compatibility shim. Phase 5 splits into (a) pure package builders,
(b) staged validator, (c) publication flow, and (d) CLI/shim/format docs.
Every sub-PR runs applicable baseline-independent tests; any producer-wiring
change also supplies fresh canonical golden evidence. Never combine extraction
with cleanup or optimization. Phase gates—not the estimate—determine pace.

**Effort:** use 17–23 engineer-days as the base implementation estimate through
Phase 6, and **22–32 engineer-days as the delivery-planning range**
(25–40% contingency) until Phase 0b establishes baseline runtime, GPU
tolerances, and fixture stability. Both ranges exclude PR queue time, GPU
scheduling delays, and optional Phase 7 work. Re-estimation is a formal Phase
0b exit artifact accepted by the engineering owner; Phase 1 does not start
until it is recorded. Phases 0a–2c deliver the reusable contract foundation in
roughly 8–10 base days and can ship standalone; the primary
monolith/readability goal is not complete until Phases 3–5 land.

## Corrections logged across nine review passes

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
- **Rev 6:** split every artifact checker into baseline-independent `validate`
  and canonical `compare-golden` modes; the normal shell uses package validation
  against current inputs and never compares arbitrary runs with scene `839920`.
- **Rev 6:** package now validates its complete staging tree before atomic
  publication; the external checker can independently confirm the final tree.
- **Rev 6:** replaced ambiguous RMSE/SSIM/T₂ language with separately named,
  directional NumPy-only RGB and depth metrics and deterministic frame selection.
- **Rev 6:** Phase 0b now uses a designated baseline, three characterization
  render pairs, committed threshold formulas, and a held-out validation pair
  (superseded by the quality-first revision 7 handoff amendment below).
- **Rev 6:** corrected `info.json` calibration assertions: intrinsic values live
  in Parquet, while `info.json` contains camera metadata plus feature schema.
- **Rev 6:** made `encoded_depth_sentinel` share the encoder's float32 multiply/
  round path and added a divergent half-step regression case.
- **Rev 6:** enumerated all shared depth fields, defined controlled immutable
  GPU re-baselining, and preserved/documented legacy coefficient/focal tolerances.
- **Rev 7:** baseline/candidate source commits and literal commands are recorded
  as audit identity, not equality-required golden fields; semantic config,
  inputs/trajectories, and relevant runtime fingerprints remain binding.
- **Rev 7:** added automatically generated candidate `run_provenance.json` and
  explicit baseline/run provenance inputs to canonical comparisons.
- **Rev 7:** Phase 0b temporarily owns the sentinel helper; Phase 1 moves it to
  `render_processing`, rewires the checker, and proves the saved numeric vectors.
- **Rev 7:** both render modes preflight the public sentinel before app/staging;
  the helper independently rejects non-finite, non-positive, and overflow inputs.
- **Rev 7:** baseline-independent checker guarantees are limited to properties
  available from current artifacts; exact/tolerant regression stays in golden mode.
- **Rev 7:** raw-depth `add` pins float32+squeeze conversion, non-mutation, scalar
  list retention, and legacy finish-time reduction order.
- **Rev 7:** canonical orchestration runs render/package golden comparisons
  independently; architecture is frozen and implementation proceeds through tickets.
- **Rev 7 handoff amendment:** runtime eligibility is stage-specific; binding
  canonical evidence requires clean Git-resolvable commits and structured,
  atomically published evidence with full ID/digest/root/status validation.
- **Rev 7 handoff amendment:** canonical trajectory digest invariance/sensitivity
  tests are exhaustive, and package input identity includes the pointcloud.
- **Rev 7 handoff amendment:** binding goldens cover every episode; tolerance
  capture uses five characterization + two held-out runs, per-metric minimum
  margins, report-only all-frame distributions, and a mutation-sensitivity suite.
- **Rev 7 handoff amendment:** publication rejects symlinks/dangling links and
  special entries; destructive shell cleanup requires numeric scenes and
  canonical strict-descendant path guards.
- **Rev 7 handoff amendment:** fixed the target-tree sentinel omission, assigned
  approval ownership, split large phases into reviewable sub-PRs, and made phase
  gates—not schedule estimates—the pace authority.
- **Rev 8:** assigned staging allocation explicitly: generate/package allocate
  internally; the orchestrator allocates one shared render stage through a
  package-safe CLI; module render requires that existing real directory; the
  legacy exact-path `os.mkdir` behavior remains a documented non-atomic
  exception.
- **Rev 8:** safely creates an absent disposable `WORK_ROOT`, requires durable
  `OUTPUT_ROOT` to be pre-provisioned, and refuses destructive targets anywhere
  under `SCRIPT_DIR`; atomic visibility no longer implies power-loss durability.
- **Rev 8:** pinned SHA-256 canonical digest framing/JSON/array/path rules and
  exact PLY identity, with committed independent test vectors.
- **Rev 8:** added atomic final `verification_manifest.json` evidence binding
  run/baseline provenance, tolerance policy, artifacts, and every checker result.
- **Rev 8:** requires independent quality-gate review by default and adds
  per-metric detection budgets, independently staged/process-isolated
  characterization runs, cache/failed-stage records, deterministic mutation
  preconditions, and pass/fail boundary mutations.
