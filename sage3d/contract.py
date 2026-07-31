"""Cross-artifact pre-package contract validator (numpy + PIL only).

Package-safe: stdlib + numpy + PIL. No Isaac, cv2, scipy, trimesh, or pxr.

Phase 2b centralizes the pre-package validation of trajectory and finalized-
render artifacts in one authority. :func:`validate_pipeline_contract` runs in
``package`` before any parquet is written and therefore cannot inspect packaged
extrinsics/calibration — those are checked on the completed staging tree by
the staged package validator/checker.

Invariants enforced (explicit exceptions, not ``assert``):

- manifest episode count ↔ npz inventory;
- manifest frame counts ↔ npz lengths;
- RGB ↔ depth frame counts;
- RGB ↔ canonical-depth calibration agreement;
- scene IDs across CLI/manifest/summaries;
- contiguous episode indexes ↔ filename stems (``naming.parse_*``);
- image inventory + dims/dtype;
- no extra/stale frames;
- manifest ``camera_height_m`` ↔ NPZ ``camera_positions[:, 2]``.

Currently-missing checks added: render modes, principal point, horizontal/
vertical FOV, forward-mask radius, explicit RGB ↔ depth agreement for all
shared depth fields (``depth_type``, ``min_depth_m``, ``max_depth_m``,
``depth_scale``), canonical depth ↔ alias equality, and per-episode summary
counts (depth only — RGB keeps ``episodes=[]``).

Float comparison policy (binding): summary ↔ summary and canonical depth ↔
alias use exact parsed value equality; manifest camera height ↔ NPZ Z uses
``np.float32`` cast then exact equality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sage3d.episode_arrays import EpisodeArrays, EPISODE_KEYS, load_episode
from sage3d.naming import (
    episode_filename,
    frame_stem,
    parse_episode_filename,
    parse_frame_filename,
)

# Shared depth fields that RGB and depth summaries must agree on even though
# RGB does not encode depth.
_SHARED_DEPTH_FIELDS = (
    "depth_type",
    "min_depth_m",
    "max_depth_m",
    "depth_scale",
)

# Calibration fields that RGB and depth summaries must agree on exactly.
_SHARED_CALIBRATION_FIELDS = (
    "resolution",
    "horizontal_fov_deg",
    "vertical_fov_deg",
    "focal_length_pixels",
    "principal_point",
    "fisheye_coefficients",
    "forward_mask_radius_pixels",
    "camera_pitch_deg",
)


class ContractError(RuntimeError):
    """Base class for cross-artifact contract violations."""


class SceneIdMismatchError(ContractError):
    """Scene IDs across CLI/manifest/summaries do not agree."""


class EpisodeCountMismatchError(ContractError):
    """Manifest episode count does not match the npz inventory."""


class FrameCountMismatchError(ContractError):
    """Manifest/summary/npz frame counts do not agree."""


class CalibrationMismatchError(ContractError):
    """RGB and canonical-depth calibration fields do not agree."""


class SharedDepthFieldMismatchError(ContractError):
    """RGB and depth summaries disagree on shared depth settings."""


class DepthAliasMismatchError(ContractError):
    """Canonical depth summary and alias summary are not equal."""


class EpisodeIndexError(ContractError):
    """Episode indexes are not contiguous or do not match filename stems."""


class ImageInventoryError(ContractError):
    """Rendered image inventory is incomplete, stale, or invalid."""


class CameraHeightMismatchError(ContractError):
    """Manifest camera_height_m does not match NPZ camera_positions[:, 2]."""


class NpzSchemaError(ContractError):
    """An episode npz has missing keys, wrong dtypes/shapes, or non-finite values."""


def _require_files(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def _check_scene_ids(
    expected_scene_id: str,
    manifest: dict[str, Any],
    rgb_summary: dict[str, Any],
    canonical_depth_summary: dict[str, Any],
    depth_alias_summary: dict[str, Any],
) -> None:
    for label, source in (
        ("manifest", manifest["scene_id"]),
        ("rgb_summary", rgb_summary["scene_id"]),
        ("canonical_depth_summary", canonical_depth_summary["scene_id"]),
        ("depth_alias_summary", depth_alias_summary["scene_id"]),
    ):
        if source != expected_scene_id:
            raise SceneIdMismatchError(
                f"scene_id mismatch: expected {expected_scene_id!r}, "
                f"{label} has {source!r}"
            )


def _check_episode_count(
    manifest: dict[str, Any],
    episodes_by_id: dict[int, EpisodeArrays],
) -> None:
    expected = manifest["episode_count"]
    actual = len(episodes_by_id)
    if actual != expected:
        raise EpisodeCountMismatchError(
            f"episode count mismatch: manifest {expected}, npz inventory {actual}"
        )
    expected_ids = set(range(expected))
    actual_ids = set(episodes_by_id)
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        raise EpisodeIndexError(
            f"episode index set mismatch: missing {sorted(missing)}, "
            f"extra {sorted(extra)}"
        )


def _check_contiguous_indexes(
    episodes_by_id: dict[int, EpisodeArrays],
    trajectory_dir: Path,
) -> None:
    """Episode indexes must be contiguous from 0 and match filename stems."""
    for episode_index in sorted(episodes_by_id):
        expected_name = episode_filename(episode_index)
        # Verify the file exists with the canonical name.
        expected_path = trajectory_dir / expected_name
        if not expected_path.is_file():
            raise EpisodeIndexError(
                f"expected episode file {expected_name} not found in {trajectory_dir}"
            )
    # Detect any extra npz files whose stems do not map to expected indexes.
    npz_files = sorted(trajectory_dir.glob("episode_*.npz"))
    for npz_file in npz_files:
        parsed = parse_episode_filename(npz_file.name)
        if parsed not in episodes_by_id:
            raise EpisodeIndexError(
                f"unexpected episode file {npz_file.name} not in manifest inventory"
            )


def _check_frame_counts_manifest_npz(
    manifest: dict[str, Any],
    episodes_by_id: dict[int, EpisodeArrays],
) -> None:
    manifest_episodes = manifest["episodes"]
    for episode_index in sorted(episodes_by_id):
        episode = episodes_by_id[episode_index]
        manifest_frame_count = manifest_episodes[episode_index]["frame_count"]
        npz_frame_count = len(episode.actions)
        if npz_frame_count != manifest_frame_count:
            raise FrameCountMismatchError(
                f"episode {episode_index}: manifest frame_count "
                f"{manifest_frame_count} != npz actions length {npz_frame_count}"
            )
        # Also verify action/point_goal internal consistency.
        if len(episode.actions) != len(episode.point_goal):
            raise FrameCountMismatchError(
                f"episode {episode_index}: actions length {len(episode.actions)} "
                f"!= point_goal length {len(episode.point_goal)}"
            )


def _check_summary_frame_counts(
    manifest: dict[str, Any],
    rgb_summary: dict[str, Any],
    canonical_depth_summary: dict[str, Any],
    depth_alias_summary: dict[str, Any],
) -> None:
    manifest_total = sum(ep["frame_count"] for ep in manifest["episodes"])
    for label, summary in (
        ("rgb_summary", rgb_summary),
        ("canonical_depth_summary", canonical_depth_summary),
        ("depth_alias_summary", depth_alias_summary),
    ):
        if summary["total_frames"] != manifest_total:
            raise FrameCountMismatchError(
                f"{label} total_frames {summary['total_frames']} != "
                f"manifest total {manifest_total}"
            )
    if rgb_summary["total_frames"] != canonical_depth_summary["total_frames"]:
        raise FrameCountMismatchError(
            f"rgb total_frames {rgb_summary['total_frames']} != "
            f"depth total_frames {canonical_depth_summary['total_frames']}"
        )


def _check_calibration_agreement(
    rgb_summary: dict[str, Any],
    canonical_depth_summary: dict[str, Any],
) -> None:
    """RGB and canonical-depth calibration fields must agree exactly."""
    for field in _SHARED_CALIBRATION_FIELDS:
        rgb_val = rgb_summary[field]
        depth_val = canonical_depth_summary[field]
        if rgb_val != depth_val:
            raise CalibrationMismatchError(
                f"calibration field {field!r} mismatch: "
                f"rgb={rgb_val!r}, depth={depth_val!r}"
            )


def _check_shared_depth_fields(
    rgb_summary: dict[str, Any],
    canonical_depth_summary: dict[str, Any],
) -> None:
    """RGB and depth must agree on shared depth settings even though RGB has no depth."""
    for field in _SHARED_DEPTH_FIELDS:
        rgb_val = rgb_summary[field]
        depth_val = canonical_depth_summary[field]
        if rgb_val != depth_val:
            raise SharedDepthFieldMismatchError(
                f"shared depth field {field!r} mismatch: "
                f"rgb={rgb_val!r}, depth={depth_val!r}"
            )


def _check_render_modes(
    rgb_summary: dict[str, Any],
    canonical_depth_summary: dict[str, Any],
) -> None:
    if rgb_summary["render_mode"] != "rgb":
        raise ContractError(
            f"rgb_summary render_mode is {rgb_summary['render_mode']!r}, expected 'rgb'"
        )
    if canonical_depth_summary["render_mode"] != "depth":
        raise ContractError(
            f"canonical_depth_summary render_mode is "
            f"{canonical_depth_summary['render_mode']!r}, expected 'depth'"
        )


def _check_depth_alias_equality(
    canonical_depth_summary: dict[str, Any],
    depth_alias_summary: dict[str, Any],
) -> None:
    """The canonical depth summary (render_summary.json) and alias
    (depth_render_summary.json) must be exactly equal."""
    if canonical_depth_summary != depth_alias_summary:
        # Find the first differing key for an actionable message.
        all_keys = set(canonical_depth_summary) | set(depth_alias_summary)
        for key in sorted(all_keys):
            if canonical_depth_summary.get(key) != depth_alias_summary.get(key):
                raise DepthAliasMismatchError(
                    f"canonical depth summary != alias summary at key {key!r}: "
                    f"render_summary={canonical_depth_summary.get(key)!r}, "
                    f"depth_render_summary={depth_alias_summary.get(key)!r}"
                )
        raise DepthAliasMismatchError(
            "canonical depth summary != alias summary (structural difference)"
        )


def _check_per_episode_summary_counts(
    manifest: dict[str, Any],
    canonical_depth_summary: dict[str, Any],
    episodes_by_id: dict[int, EpisodeArrays],
) -> None:
    """Depth per-episode summary counts must match manifest/npz; RGB keeps
    episodes=[]."""
    summary_episodes = canonical_depth_summary["episodes"]
    if len(summary_episodes) != len(episodes_by_id):
        raise FrameCountMismatchError(
            f"depth summary has {len(summary_episodes)} episode records, "
            f"expected {len(episodes_by_id)}"
        )
    for summary_ep in summary_episodes:
        idx = summary_ep["episode_index"]
        if idx not in episodes_by_id:
            raise EpisodeIndexError(
                f"depth summary references episode {idx} not in npz inventory"
            )
        npz_count = len(episodes_by_id[idx].actions)
        if summary_ep["frame_count"] != npz_count:
            raise FrameCountMismatchError(
                f"depth summary episode {idx} frame_count "
                f"{summary_ep['frame_count']} != npz {npz_count}"
            )
        manifest_count = manifest["episodes"][idx]["frame_count"]
        if summary_ep["frame_count"] != manifest_count:
            raise FrameCountMismatchError(
                f"depth summary episode {idx} frame_count "
                f"{summary_ep['frame_count']} != manifest {manifest_count}"
            )


def _check_camera_height(
    manifest: dict[str, Any],
    episodes_by_id: dict[int, EpisodeArrays],
) -> None:
    """Manifest camera_height_m must exactly match every NPZ camera_positions[:, 2]
    after casting to float32."""
    expected_z = np.float32(manifest["camera_height_m"])
    for episode_index in sorted(episodes_by_id):
        episode = episodes_by_id[episode_index]
        z_values = episode.camera_positions[:, 2].astype(np.float32)
        if not np.all(z_values == expected_z):
            raise CameraHeightMismatchError(
                f"episode {episode_index}: manifest camera_height_m "
                f"{manifest['camera_height_m']!r} does not match all NPZ "
                f"camera_positions[:, 2] (unique={np.unique(z_values).tolist()})"
            )


def _check_npz_schema(
    episodes_by_id: dict[int, EpisodeArrays],
    trajectory_dir: Path,
) -> None:
    """Re-validate every episode npz for keys, dtypes, shapes, finiteness, and
    a common frame length using allow_pickle=False in a context manager."""
    for episode_index in sorted(episodes_by_id):
        path = trajectory_dir / episode_filename(episode_index)
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            missing = [k for k in EPISODE_KEYS if k not in keys]
            if missing:
                raise NpzSchemaError(
                    f"{path} missing keys: {missing}"
                )
            extra = keys - set(EPISODE_KEYS)
            if extra:
                raise NpzSchemaError(
                    f"{path} has extra keys: {sorted(extra)}"
                )
            arrays = {k: data[k] for k in EPISODE_KEYS}
        # Dtype checks.
        for key in ("points", "actions", "camera_positions", "yaw",
                    "point_goal", "start_position", "goal_position"):
            if arrays[key].dtype != np.float32:
                raise NpzSchemaError(
                    f"{path} key {key!r} dtype {arrays[key].dtype} != float32"
                )
        # Shape checks.
        n = arrays["actions"].shape[0]
        for key in ("camera_positions", "yaw", "point_goal"):
            if arrays[key].shape[0] != n:
                raise NpzSchemaError(
                    f"{path} key {key!r} leading dim {arrays[key].shape[0]} "
                    f"!= actions leading dim {n}"
                )
        if arrays["actions"].shape != (n, 4, 4):
            raise NpzSchemaError(
                f"{path} actions shape {arrays['actions'].shape} != ({n}, 4, 4)"
            )
        if arrays["camera_positions"].shape != (n, 3):
            raise NpzSchemaError(
                f"{path} camera_positions shape {arrays['camera_positions'].shape} "
                f"!= ({n}, 3)"
            )
        if arrays["point_goal"].shape != (n, 2):
            raise NpzSchemaError(
                f"{path} point_goal shape {arrays['point_goal'].shape} != ({n}, 2)"
            )
        if arrays["yaw"].shape != (n,):
            raise NpzSchemaError(
                f"{path} yaw shape {arrays['yaw'].shape} != ({n},)"
            )
        if arrays["start_position"].shape != (3,):
            raise NpzSchemaError(
                f"{path} start_position shape {arrays['start_position'].shape} != (3,)"
            )
        if arrays["goal_position"].shape != (3,):
            raise NpzSchemaError(
                f"{path} goal_position shape {arrays['goal_position'].shape} != (3,)"
            )
        if arrays["points"].shape[0] != n:
            raise NpzSchemaError(
                f"{path} points leading dim {arrays['points'].shape[0]} "
                f"!= actions leading dim {n}"
            )
        # Finiteness checks for all float arrays.
        for key in EPISODE_KEYS:
            if not np.all(np.isfinite(arrays[key])):
                raise NpzSchemaError(
                    f"{path} key {key!r} has non-finite values"
                )


def _check_image_inventory(
    episodes_by_id: dict[int, EpisodeArrays],
    rendered_dir: Path,
    width: int,
    height: int,
) -> None:
    """Verify every expected RGB/depth frame exists, has correct dims/dtype,
    and that there are no extra or stale frames."""
    rgb_dir = rendered_dir / "observation.images.rgb"
    depth_dir = rendered_dir / "observation.images.depth"
    if not rgb_dir.is_dir():
        raise ImageInventoryError(f"rgb image directory not found: {rgb_dir}")
    if not depth_dir.is_dir():
        raise ImageInventoryError(f"depth image directory not found: {depth_dir}")

    expected_stems: set[str] = set()
    for episode_index in sorted(episodes_by_id):
        frame_count = len(episodes_by_id[episode_index].actions)
        for frame_index in range(frame_count):
            expected_stems.add(frame_stem(episode_index, frame_index))

    rgb_files = {p.stem for p in rgb_dir.glob("*.jpg")}
    depth_files = {p.stem for p in depth_dir.glob("*.png")}

    # Missing files.
    missing_rgb = expected_stems - rgb_files
    if missing_rgb:
        raise ImageInventoryError(
            f"missing {len(missing_rgb)} rgb frames, e.g. {sorted(missing_rgb)[:3]}"
        )
    missing_depth = expected_stems - depth_files
    if missing_depth:
        raise ImageInventoryError(
            f"missing {len(missing_depth)} depth frames, e.g. {sorted(missing_depth)[:3]}"
        )

    # Stale / extra files.
    extra_rgb = rgb_files - expected_stems
    if extra_rgb:
        raise ImageInventoryError(
            f"{len(extra_rgb)} stale/extra rgb frames, e.g. {sorted(extra_rgb)[:3]}"
        )
    extra_depth = depth_files - expected_stems
    if extra_depth:
        raise ImageInventoryError(
            f"{len(extra_depth)} stale/extra depth frames, e.g. {sorted(extra_depth)[:3]}"
        )

    # Validate filename stems parse correctly and indexes are within bounds.
    for stem in rgb_files | depth_files:
        ep_idx, frame_idx = parse_frame_filename(stem)
        if ep_idx not in episodes_by_id:
            raise ImageInventoryError(
                f"rgb/depth stem {stem!r} references unknown episode {ep_idx}"
            )
        if frame_idx >= len(episodes_by_id[ep_idx].actions):
            raise ImageInventoryError(
                f"rgb/depth stem {stem!r} frame index {frame_idx} out of range "
                f"for episode {ep_idx} ({len(episodes_by_id[ep_idx].actions)} frames)"
            )

    # Validate dims/dtype of each image.
    for stem in sorted(expected_stems):
        rgb_path = rgb_dir / f"{stem}.jpg"
        with Image.open(rgb_path) as rgb:
            if rgb.size != (width, height):
                raise ImageInventoryError(
                    f"rgb {rgb_path.name}: size {rgb.size} != ({width}, {height})"
                )
            if rgb.mode != "RGB":
                raise ImageInventoryError(
                    f"rgb {rgb_path.name}: mode {rgb.mode!r} != 'RGB'"
                )
        depth_path = depth_dir / f"{stem}.png"
        with Image.open(depth_path) as depth:
            depth_array = np.asarray(depth)
            if depth.size != (width, height):
                raise ImageInventoryError(
                    f"depth {depth_path.name}: size {depth.size} != ({width}, {height})"
                )
            if depth_array.dtype != np.uint16:
                raise ImageInventoryError(
                    f"depth {depth_path.name}: dtype {depth_array.dtype} != uint16"
                )


def validate_pipeline_contract(
    expected_scene_id: str,
    manifest: dict[str, Any],
    rgb_summary: dict[str, Any],
    canonical_depth_summary: dict[str, Any],
    depth_alias_summary: dict[str, Any],
    episodes_by_id: dict[int, EpisodeArrays],
    trajectory_dir: Path,
    rendered_dir: Path,
    pointcloud_path: Path,
) -> None:
    """Validate all pre-package cross-artifact invariants.

    Runs in ``package`` before any parquet is written. Raises a subclass of
    :class:`ContractError` (or ``FileNotFoundError``) with an actionable message
    on the first violation.
    """
    _require_files(
        (pointcloud_path,)
    )

    _check_scene_ids(
        expected_scene_id,
        manifest,
        rgb_summary,
        canonical_depth_summary,
        depth_alias_summary,
    )
    _check_render_modes(rgb_summary, canonical_depth_summary)
    _check_calibration_agreement(rgb_summary, canonical_depth_summary)
    _check_shared_depth_fields(rgb_summary, canonical_depth_summary)
    _check_depth_alias_equality(canonical_depth_summary, depth_alias_summary)
    _check_episode_count(manifest, episodes_by_id)
    _check_contiguous_indexes(episodes_by_id, trajectory_dir)
    _check_frame_counts_manifest_npz(manifest, episodes_by_id)
    _check_summary_frame_counts(
        manifest, rgb_summary, canonical_depth_summary, depth_alias_summary
    )
    _check_per_episode_summary_counts(manifest, canonical_depth_summary, episodes_by_id)
    _check_camera_height(manifest, episodes_by_id)
    _check_npz_schema(episodes_by_id, trajectory_dir)

    width, height = canonical_depth_summary["resolution"]
    _check_image_inventory(episodes_by_id, rendered_dir, width, height)