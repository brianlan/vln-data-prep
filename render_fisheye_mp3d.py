import blenderproc as bproc  # blenderproc must be imported before bpy

"""Render equidistant fisheye RGB and depth from textured MP3D OBJ scenes.

Scene-N1's ``mp3d_n1`` assets contain one Matterport OBJ/MTL and a set of JPEG
texture tiles per scene. Both the raw OBJ vertices and InternData-N1 action
matrices use the same Z-up world frame. Blender's OBJ importer is configured to
preserve that frame, and the actions are used directly as camera-to-world poses.
"""

import argparse
import os

import numpy as np
from PIL import Image


def load_actions(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        return data["actions"]


def import_textured_obj(filepath):
    """Import a Z-up MP3D OBJ without changing its stored coordinates."""
    import bpy

    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=filepath, forward_axis="Y", up_axis="Z")
    return [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]


def make_materials_emissive(mesh_objects):
    """Preserve captured texture colors without applying artificial lighting."""
    materials = []
    seen = set()
    for obj in mesh_objects:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is not None and mat.as_pointer() not in seen:
                seen.add(mat.as_pointer())
                materials.append(mat)

    textured = 0
    converted = 0
    for mat in materials:
        mat.use_nodes = True
        mat.use_backface_culling = False
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        output = next(
            (node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output),
            None,
        )
        if output is None:
            output = nodes.new("ShaderNodeOutputMaterial")

        emission = nodes.new("ShaderNodeEmission")
        emission.name = "MP3D Captured Appearance"
        emission.inputs["Strength"].default_value = 1.0

        if bsdf is not None and bsdf.inputs["Base Color"].is_linked:
            source = bsdf.inputs["Base Color"].links[0].from_socket
            links.new(source, emission.inputs["Color"])
            if any(node.type == "TEX_IMAGE" and node.image is not None for node in nodes):
                textured += 1
        elif bsdf is not None:
            emission.inputs["Color"].default_value = tuple(
                bsdf.inputs["Base Color"].default_value
            )
        else:
            emission.inputs["Color"].default_value = tuple(mat.diffuse_color)

        for link in list(output.inputs["Surface"].links):
            links.remove(link)
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        converted += 1

    if not materials:
        raise RuntimeError("MP3D OBJ imported without any materials")
    if textured == 0:
        raise RuntimeError(
            "MP3D materials loaded, but no diffuse texture image was found. "
            "Check that the OBJ, MTL, and JPEG tiles are colocated."
        )
    print(f"[render] Materials: {converted} converted, {textured} textured")


def object_bounds(mesh_objects):
    """Return a world-space AABB across all imported mesh objects."""
    from mathutils import Vector

    corners = []
    for obj in mesh_objects:
        corners.extend(np.asarray(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box)
    points = np.asarray(corners, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def validate_pose_bounds(actions_by_episode, bounds_min, bounds_max, margin=0.5):
    """Fail early when scene identity or axis conventions do not match."""
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
            f"{outside.sum()}/{len(positions)} camera positions are outside MP3D bounds; "
            f"first={sample}, bounds={bounds_min}..{bounds_max}. "
            "This usually indicates a scene-ID or axis-convention mismatch."
        )
    print(
        f"[render] Pose/bounds check OK: {len(positions)} cameras inside "
        f"{bounds_min} .. {bounds_max}"
    )


def setup_world():
    import bpy

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        background.inputs["Strength"].default_value = 1.0


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mesh", required=True, help="Path to the textured MP3D OBJ")
    parser.add_argument("--traj_dir", required=True, help="Prepared episode NPZ directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov_deg", type=float, default=195.0)
    parser.add_argument("--depth_scale", type=float, default=10000.0)
    parser.add_argument("--max_depth_m", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--max_episodes", type=int, default=0,
        help="Render only the first N episodes; 0 renders all",
    )
    parser.add_argument(
        "--frame_stride", type=int, default=1,
        help="Render every Nth frame (for smoke tests)",
    )
    args, _ = parser.parse_known_args()
    if args.frame_stride < 1:
        parser.error("--frame_stride must be >= 1")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"[render] Scene={args.scene}, mesh={args.mesh}, FOV={args.fov_deg}deg, "
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

    print(f"[render] Importing textured MP3D OBJ: {args.mesh}")
    mesh_objects = import_textured_obj(args.mesh)
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {args.mesh}")
    make_materials_emissive(mesh_objects)
    bounds_min, bounds_max = object_bounds(mesh_objects)
    print(f"[render] Imported {len(mesh_objects)} mesh object(s), bounds={bounds_min}..{bounds_max}")

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

    episode_info = []
    total_frames = 0
    for ep_idx, (name, actions) in enumerate(zip(npz_files, actions_by_episode)):
        frame_indices = list(range(0, len(actions), args.frame_stride))
        episode_info.append((ep_idx, frame_indices))
        for frame_idx in frame_indices:
            bproc.camera.add_camera_pose(actions[frame_idx])
        total_frames += len(frame_indices)
        print(f"[render] Episode {ep_idx}: {len(frame_indices)} frames ({name})")

    print(f"[render] Rendering {total_frames} total frames")
    data = bproc.renderer.render()

    rgb_dir = os.path.join(args.output_dir, "observation.images.rgb")
    depth_dir = os.path.join(args.output_dir, "observation.images.depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    output_idx = 0
    for ep_idx, frame_indices in episode_info:
        for frame_idx in frame_indices:
            Image.fromarray(data["colors"][output_idx]).save(
                os.path.join(rgb_dir, f"episode_{ep_idx:06d}_{frame_idx:03d}.jpg"),
                quality=95,
            )
            depth_m = np.nan_to_num(
                data["depth"][output_idx],
                nan=args.max_depth_m,
                posinf=args.max_depth_m,
                neginf=0.0,
            )
            depth_m = np.clip(depth_m, 0.0, args.max_depth_m)
            depth_u16 = (depth_m * args.depth_scale).astype(np.uint16)
            Image.fromarray(depth_u16, mode="I;16").save(
                os.path.join(depth_dir, f"episode_{ep_idx:06d}_{frame_idx:03d}.png")
            )
            output_idx += 1

    print(f"[render] Saved {output_idx} RGB/depth pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
