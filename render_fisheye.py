import blenderproc as bproc  # MUST be first import

"""
BlenderProc script: render fisheye RGB+depth from Replica mesh + trajectory poses.

Usage:
    blenderproc run render_fisheye.py --scene apartment_1 --mesh /ssd5/datasets/Replica/apartment_0/mesh.ply ...

Reads trajectory parquet files, computes camera-to-world poses, loads the Replica
mesh with vertex colors, and renders fisheye (equidistant) RGB + depth images.
"""

import argparse
import json
import os
import sys
import shutil

import numpy as np
from PIL import Image


def load_trajectory(npz_path):
    """Load pre-extracted trajectory .npz, return (extrinsic, actions)."""
    data = np.load(npz_path, allow_pickle=False)
    return data["extrinsic"], data["actions"]


def setup_fisheye_camera(width, height, fov_rad):
    """
    Configure the Blender camera as equidistant fisheye.
    Returns the camera object.
    """
    import bpy

    cam_data = bpy.data.cameras.new(name="FisheyeCamera")
    cam_data.type = "PANO"
    cam_data.panorama_type = "FISHEYE_EQUIDISTANT"
    cam_data.fisheye_fov = fov_rad
    cam_obj = bpy.data.objects.new("FisheyeCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    return cam_obj


def setup_mesh_material(mesh_obj):
    """Create a material that uses vertex colors as emissive (pre-baked lighting)."""
    import bpy

    raw_obj = mesh_obj.blender_obj if hasattr(mesh_obj, "blender_obj") else mesh_obj
    mesh_data = raw_obj.data

    # Find the color attribute layer name (Blender 4.x uses color_attributes)
    color_attr_name = None
    if hasattr(mesh_data, "color_attributes") and mesh_data.color_attributes:
        for layer in mesh_data.color_attributes:
            print(f"[render] Found color attribute: '{layer.name}' (domain={layer.domain}, type={layer.data_type})")
            color_attr_name = layer.name
            break

    mat = bpy.data.materials.new(name="VertexColorMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output_node = nodes.new("ShaderNodeOutputMaterial")

    # Use Emission shader with vertex color as emission color.
    # Replica vertex colors are pre-baked with lighting, so the mesh is self-illuminating.
    # This avoids needing scene lights and produces correct colors.
    emission = nodes.new("ShaderNodeEmission")

    if color_attr_name is not None:
        # Use Vertex Color node (works with color_attributes in Blender 4.x)
        vc_node = nodes.new("ShaderNodeVertexColor")
        vc_node.layer_name = color_attr_name
        links.new(vc_node.outputs["Color"], emission.inputs["Color"])
        print(f"[render] Using color attribute '{color_attr_name}' with Emission shader")
    else:
        # Fallback: try "Col" layer name
        vc_node = nodes.new("ShaderNodeVertexColor")
        vc_node.layer_name = "Col"
        links.new(vc_node.outputs["Color"], emission.inputs["Color"])
        print("[render] WARNING: Using fallback Vertex Color node with layer 'Col'")

    links.new(emission.outputs["Emission"], output_node.inputs["Surface"])

    raw_obj.data.materials.clear()
    raw_obj.data.materials.append(mat)
    return mat


def setup_lighting():
    """Set dark world background (mesh is emissive, no scene lights needed)."""
    import bpy

    # Set world background to dark (mesh provides its own lighting via emission)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Scene name, e.g. apartment_1")
    parser.add_argument("--mesh", required=True, help="Path to Replica mesh.ply")
    parser.add_argument("--traj_dir", required=True, help="Dir with pre-extracted .npy trajectory files")
    parser.add_argument("--output_dir", required=True, help="Where to save rendered images")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov_deg", type=float, default=195.0, help="Fisheye FOV in degrees")
    parser.add_argument("--depth_scale", type=float, default=10000.0,
                        help="Multiply meters by this to get uint16 depth values")
    parser.add_argument("--max_depth_m", type=float, default=6.0,
                        help="Clip depth beyond this (meters)")
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fov_rad = np.radians(args.fov_deg)
    print(f"[render] Scene={args.scene}, FOV={args.fov_deg}° ({fov_rad:.4f} rad), "
          f"resolution={args.width}x{args.height}")

    # --- Init BlenderProc ---
    bproc.init()
    import bpy

    # Use Cycles (required for fisheye panorama)
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 16  # low samples for speed
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = False

    # Use Standard view transform (Filmic darkens pre-baked vertex colors)
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1.0

    # --- Load mesh ---
    print(f"[render] Loading mesh: {args.mesh}")
    # Use Blender's built-in PLY importer (Blender 4.x: wm.ply_import)
    bpy.ops.wm.ply_import(filepath=args.mesh)
    # The imported object is the last added mesh object
    mesh_objs = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    raw_mesh = mesh_objs[-1]  # most recently added
    # Wrap in MeshObject for BlenderProc compatibility
    mesh_obj = bproc.types.MeshObject(raw_mesh)
    setup_mesh_material(mesh_obj)
    print(f"[render] Mesh loaded: {len(raw_mesh.data.vertices)} vertices, "
          f"{len(raw_mesh.data.polygons)} faces")

    # Check vertex color layers (Blender 4.x uses color_attributes)
    if hasattr(raw_mesh.data, "color_attributes") and raw_mesh.data.color_attributes:
        print(f"[render] Color attributes: {[l.name for l in raw_mesh.data.color_attributes]}")
    elif raw_mesh.data.vertex_colors:
        print(f"[render] Vertex color layers (legacy): {[l.name for l in raw_mesh.data.vertex_colors]}")
    else:
        print("[render] WARNING: No vertex color layers found!")

    # --- Lighting ---
    setup_lighting()

    # --- Fisheye camera ---
    cam_obj = setup_fisheye_camera(args.width, args.height, fov_rad)
    print(f"[render] Fisheye camera configured: equidistant, fov={args.fov_deg}°")

    # --- Enable depth output ---
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # --- Find all episode .npz files ---
    npy_files = sorted([f for f in os.listdir(args.traj_dir) if f.endswith(".npz")])
    print(f"[render] Found {len(npy_files)} episode trajectory files")

    # --- Load all trajectories and add all camera poses at once ---
    episode_info = []  # (ep_idx, n_frames) for splitting results
    total_frames = 0
    for ep_idx, npy_file in enumerate(npy_files):
        npy_path = os.path.join(args.traj_dir, npy_file)
        extrinsic, actions = load_trajectory(npy_path)
        n_frames = actions.shape[0]
        episode_info.append((ep_idx, n_frames))
        print(f"[render] Episode {ep_idx}: {n_frames} frames, file={npy_file}")

        for i in range(n_frames):
            cam2world = actions[i]  # action IS the camera-to-world pose directly
            bproc.camera.add_camera_pose(cam2world)
        total_frames += n_frames

    print(f"[render] Total frames to render: {total_frames}")

    # --- Render all frames at once ---
    print(f"[render] Starting render...")
    data = bproc.renderer.render()
    print(f"[render] Render complete!")

    # --- Save RGB and depth ---
    rgb_dir = os.path.join(args.output_dir, "observation.images.rgb")
    depth_dir = os.path.join(args.output_dir, "observation.images.depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rgb_images = data["colors"]  # list of HxWx3 uint8 arrays
    depth_images = data["depth"]  # list of HxW float32 arrays (meters)

    frame_offset = 0
    for ep_idx, n_frames in episode_info:
        for i in range(n_frames):
            fi = frame_offset + i
            # RGB
            rgb = rgb_images[fi]
            rgb_path = os.path.join(rgb_dir, f"episode_{ep_idx:06d}_{i:03d}.jpg")
            Image.fromarray(rgb).save(rgb_path, quality=95)

            # Depth: meters -> uint16 (meters * depth_scale), clip
            depth_m = depth_images[fi]
            depth_m = np.clip(depth_m, 0, args.max_depth_m)
            depth_u16 = (depth_m * args.depth_scale).astype(np.uint16)
            depth_path = os.path.join(depth_dir, f"episode_{ep_idx:06d}_{i:03d}.png")
            Image.fromarray(depth_u16, mode="I;16").save(depth_path)

        print(f"[render] Episode {ep_idx}: saved {n_frames} RGB+depth frames")
        frame_offset += n_frames

    print(f"[render] Done! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
