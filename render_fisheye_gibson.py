import blenderproc as bproc  # blenderproc must be imported before bpy

"""Render equidistant fisheye RGB and depth from textured Gibson V2 scenes.

The Gibson V2 distribution contains two meshes per scene:

* ``<scene>_mesh_texture.obj``: high-detail UV-textured render mesh (use this)
* ``mesh_z_up.obj``: simplified collision/traversability mesh (do not use for RGB)

Both the textured OBJ and InternData-N1 ``action`` matrices are already Z-up and
share the same world frame. Blender's OBJ importer is therefore configured with
``forward_axis=Y, up_axis=Z``. For Blender 4.2's importer this preserves the raw
OBJ coordinates; selecting ``-Y`` introduces an unwanted 180-degree Z rotation.
"""

import argparse
import os

import numpy as np
from PIL import Image


def load_trajectory(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    return data["actions"]


def import_textured_obj(filepath):
    """Import a Z-up OBJ without axis conversion and return new mesh objects."""
    import bpy

    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(
        filepath=filepath,
        forward_axis="Y",
        up_axis="Z",
    )
    return [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]


def make_materials_emissive(mesh_objects):
    """Preserve each diffuse texture while removing dependence on scene lights.

    Gibson's atlas texture contains the captured scene appearance. Treating it as
    emission avoids applying a second, artificial lighting pass. It also overrides
    the legacy MTL ``Tr 1`` value, which some importers interpret as transparent.
    """
    import bpy

    materials = []
    seen = set()
    for obj in mesh_objects:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is not None and mat.name not in seen:
                seen.add(mat.name)
                materials.append(mat)

    textured = 0
    for mat in materials:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        image = None
        fallback_color = (0.8, 0.8, 0.8, 1.0)
        for node in nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                image = node.image
                break
            if node.type == "BSDF_PRINCIPLED":
                fallback_color = tuple(node.inputs["Base Color"].default_value)

        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0

        if image is not None:
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = image
            tex.interpolation = "Linear"
            links.new(tex.outputs["Color"], emission.inputs["Color"])
            textured += 1
        else:
            emission.inputs["Color"].default_value = fallback_color

        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        mat.diffuse_color = (*fallback_color[:3], 1.0)

    if not materials:
        raise RuntimeError("Gibson OBJ imported without any materials")
    if textured == 0:
        raise RuntimeError(
            "Gibson materials loaded, but no diffuse texture image was found. "
            "Check that the .obj.mtl and texture PNG are next to the OBJ."
        )

    print(f"[render] Materials: {len(materials)} total, {textured} textured/emissive")


def object_bounds(mesh_objects):
    """Return world-space AABB across all imported mesh objects."""
    from mathutils import Vector

    corners = []
    for obj in mesh_objects:
        corners.extend(np.asarray(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box)
    points = np.asarray(corners, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def validate_pose_bounds(actions, bounds_min, bounds_max, margin=0.5):
    """Catch accidental coordinate/axis conversion before an expensive render."""
    positions = np.concatenate([a[:, :3, 3] for a in actions], axis=0)
    outside = np.any(
        (positions < bounds_min[None, :] - margin)
        | (positions > bounds_max[None, :] + margin),
        axis=1,
    )
    if outside.any():
        sample = positions[outside][0]
        raise RuntimeError(
            f"{outside.sum()}/{len(positions)} camera positions are outside the mesh bounds; "
            f"first={sample}, bounds={bounds_min}..{bounds_max}. "
            "This usually indicates an axis-convention mismatch."
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
    parser.add_argument("--mesh", required=True, help="Path to <scene>_mesh_texture.obj")
    parser.add_argument("--traj_dir", required=True, help="Directory of prepared episode NPZ files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov_deg", type=float, default=195.0)
    parser.add_argument("--depth_scale", type=float, default=10000.0)
    parser.add_argument("--max_depth_m", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Render only the first N episodes; 0 renders all")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Render every Nth frame (for smoke tests)")
    args, _ = parser.parse_known_args()
    if args.frame_stride < 1:
        parser.error("--frame_stride must be >= 1")
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

    print(f"[render] Importing textured OBJ: {args.mesh}")
    mesh_objects = import_textured_obj(args.mesh)
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {args.mesh}")
    make_materials_emissive(mesh_objects)
    bounds_min, bounds_max = object_bounds(mesh_objects)
    print(f"[render] Imported {len(mesh_objects)} mesh object(s), bounds={bounds_min}..{bounds_max}")

    npz_files = sorted(f for f in os.listdir(args.traj_dir) if f.endswith(".npz"))
    if args.max_episodes > 0:
        npz_files = npz_files[: args.max_episodes]
    if not npz_files:
        raise RuntimeError(f"No trajectory NPZ files found in {args.traj_dir}")

    actions_by_episode = [load_trajectory(os.path.join(args.traj_dir, name)) for name in npz_files]
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
            rgb = data["colors"][output_idx]
            Image.fromarray(rgb).save(
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
