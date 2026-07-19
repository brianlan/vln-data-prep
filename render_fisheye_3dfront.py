import blenderproc as bproc  # blenderproc must be imported before bpy

"""Render equidistant fisheye RGB and depth from a 3D-FRONT scene.

BlenderProc's official loader assembles architectural meshes from the scene
JSON, imports the referenced 3D-FUTURE furniture, and converts the source Y-up
coordinates to Blender Z-up. InternData-N1 action matrices are already in that
same Z-up world frame and are therefore used directly as camera-to-world poses.
"""

import argparse
import os

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene_json", required=True)
    parser.add_argument("--future_model_path", required=True)
    parser.add_argument("--front_texture_path", required=True)
    parser.add_argument("--traj_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov_deg", type=float, default=195.0)
    parser.add_argument("--depth_scale", type=float, default=10000.0)
    parser.add_argument("--max_depth_m", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    args, _ = parser.parse_known_args()
    if args.frame_stride < 1:
        parser.error("--frame_stride must be >= 1")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    return args


def load_actions(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        return data["actions"]


def object_bounds(mesh_objects):
    from mathutils import Vector

    corners = []
    for obj in mesh_objects:
        corners.extend(np.asarray(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box)
    points = np.asarray(corners, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def validate_pose_bounds(actions_by_episode, bounds_min, bounds_max, margin=0.5):
    positions = np.concatenate(
        [actions[:, :3, 3] for actions in actions_by_episode], axis=0
    )
    outside = np.any(
        (positions < bounds_min[None, :] - margin)
        | (positions > bounds_max[None, :] + margin),
        axis=1,
    )
    if outside.any():
        sample = positions[outside][0]
        raise RuntimeError(
            f"{outside.sum()}/{len(positions)} camera positions are outside "
            f"3D-FRONT bounds; first={sample}, bounds={bounds_min}..{bounds_max}. "
            "This usually indicates a scene-ID or axis-convention mismatch."
        )
    print(
        f"[render] Pose/bounds check OK: {len(positions)} cameras inside "
        f"{bounds_min} .. {bounds_max}"
    )


def fix_materials():
    """Avoid legacy OBJ transparency/emission while retaining synthetic PBR."""
    import bpy

    fixed = 0
    textured = 0
    for material in bpy.data.materials:
        material.use_backface_culling = False
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            if "Metallic" in node.inputs:
                node.inputs["Metallic"].default_value = 0.0
            if "Transmission Weight" in node.inputs:
                node.inputs["Transmission Weight"].default_value = 0.0
            if "Alpha" in node.inputs and not node.inputs["Alpha"].is_linked:
                node.inputs["Alpha"].default_value = 1.0
            if node.inputs["Base Color"].is_linked:
                textured += 1
            fixed += 1
    print(f"[render] Materials: {fixed} Principled shaders fixed, {textured} textured")


def setup_world():
    import bpy

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        background.inputs["Strength"].default_value = 0.35


def setup_fisheye_camera(width, height, fov_rad):
    import bpy

    camera_data = bpy.data.cameras.new(name="FisheyeCamera")
    camera_data.type = "PANO"
    camera_data.panorama_type = "FISHEYE_EQUIDISTANT"
    camera_data.fisheye_fov = fov_rad
    camera = bpy.data.objects.new("FisheyeCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"[render] Scene={args.scene}, FOV={args.fov_deg}deg, "
        f"resolution={args.width}x{args.height}, samples={args.samples}"
    )

    bproc.init()
    import bpy

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.view_settings.exposure = 0.0
    bpy.context.scene.view_settings.gamma = 1.0

    mapping_path = bproc.utility.resolve_resource("front_3D/3D_front_mapping.csv")
    label_mapping = bproc.utility.LabelIdMapping.from_csv(mapping_path)
    print(f"[render] Loading 3D-FRONT JSON: {args.scene_json}")
    bproc.loader.load_front3d(
        json_path=args.scene_json,
        future_model_path=args.future_model_path,
        front_3D_texture_path=args.front_texture_path,
        label_mapping=label_mapping,
        ceiling_light_strength=2.0,
        lamp_light_strength=10.0,
    )
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects loaded for {args.scene}")
    fix_materials()
    bounds_min, bounds_max = object_bounds(mesh_objects)
    print(
        f"[render] Loaded {len(mesh_objects)} mesh object(s), "
        f"bounds={bounds_min}..{bounds_max}"
    )

    npz_files = sorted(name for name in os.listdir(args.traj_dir) if name.endswith(".npz"))
    if args.max_episodes > 0:
        npz_files = npz_files[: args.max_episodes]
    if not npz_files:
        raise RuntimeError(f"No trajectory NPZ files found in {args.traj_dir}")
    actions_by_episode = [
        load_actions(os.path.join(args.traj_dir, name)) for name in npz_files
    ]
    validate_pose_bounds(actions_by_episode, bounds_min, bounds_max)

    setup_world()
    setup_fisheye_camera(args.width, args.height, np.radians(args.fov_deg))
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    rgb_dir = os.path.join(args.output_dir, "observation.images.rgb")
    depth_dir = os.path.join(args.output_dir, "observation.images.depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    episode_info = [
        (ep_idx, name, actions, list(range(0, len(actions), args.frame_stride)))
        for ep_idx, (name, actions) in enumerate(zip(npz_files, actions_by_episode))
    ]
    total_frames = sum(len(frame_indices) for _, _, _, frame_indices in episode_info)
    print(
        f"[render] Rendering {total_frames} total frames in "
        f"{len(episode_info)} episode-sized batches"
    )

    saved_frames = 0
    for ep_idx, name, actions, frame_indices in episode_info:
        # Rendering thousands of poses in one call retains every float RGB/depth
        # array and temporary EXR at once. Episode-sized batches keep memory and
        # /dev/shm bounded while preserving the original frame names.
        bproc.utility.reset_keyframes()
        for frame_idx in frame_indices:
            bproc.camera.add_camera_pose(actions[frame_idx])
        print(f"[render] Episode {ep_idx}: rendering {len(frame_indices)} frames ({name})")
        data = bproc.renderer.render(output_key="colors" if ep_idx == 0 else None)

        for batch_idx, frame_idx in enumerate(frame_indices):
            Image.fromarray(data["colors"][batch_idx]).save(
                os.path.join(rgb_dir, f"episode_{ep_idx:06d}_{frame_idx:03d}.jpg"),
                quality=95,
            )
            depth_m = np.nan_to_num(
                data["depth"][batch_idx],
                nan=args.max_depth_m,
                posinf=args.max_depth_m,
                neginf=0.0,
            )
            depth_m = np.clip(depth_m, 0.0, args.max_depth_m)
            depth_u16 = (depth_m * args.depth_scale).astype(np.uint16)
            Image.fromarray(depth_u16, mode="I;16").save(
                os.path.join(depth_dir, f"episode_{ep_idx:06d}_{frame_idx:03d}.png")
            )
            saved_frames += 1

    print(f"[render] Saved {saved_frames} RGB/depth pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
