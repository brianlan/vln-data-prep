import blenderproc as bproc  # blenderproc must precede other imports

"""
BlenderProc script: render fisheye RGB+depth from an HSSD-hab composite scene
+ trajectory poses (hssd_zed).

Unlike Replica (single self-contained mesh.ply with pre-baked vertex colors),
HSSD scenes are composite:
  - 1 stage GLB (the architectural shell)
  - N object instances, each = an objects/<hash>.glb placed by
    translation + rotation quaternion (w,x,y,z) + non_uniform_scale

HSSD GLBs ship real PBR textures (no baked lighting), so we render with the
imported materials and add scene lighting (world ambient + sun), NOT the
Replica emission/vertex-color shader.

Coordinate notes (validated empirically + Habitat docs):
  - HSSD stage_config: up=[0,1,0] (Y-up), front=[0,0,-1]  -> glTF convention.
  - bpy.ops.import_scene.gltf auto-converts glTF Y-up -> Blender Z-up.
  - The hssd_zed trajectory poses are Z-up (camera Z stays ~1.02m constant),
    so action[i] is used directly as the Blender cam2world, same as Replica.

Usage:
    blenderproc run render_fisheye_hssd.py \
        --scene 102344049 \
        --scene_instance /ssd5/datasets/hssd-hab/scenes/102344049.scene_instance.json \
        --hssd_root /ssd5/datasets/hssd-hab \
        --traj_dir <npz dir> --output_dir <out> [--stage_only]
"""

import argparse
import json
import os
import glob

import numpy as np
from PIL import Image


def load_trajectory(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    return data["extrinsic"], data["actions"]


def resolve_object_glb(hssd_root, template_name, glb_cache=None):
    if glb_cache:
        cached = os.path.join(glb_cache, template_name + ".glb")
        if os.path.exists(cached):
            return cached
    cand = os.path.join(hssd_root, "objects", template_name[0], template_name + ".glb")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join(hssd_root, "objects", "**", template_name + ".glb"),
                     recursive=True)
    hits = [h for h in hits if "collider" not in os.path.basename(h)]
    return hits[0] if hits else None


def resolve_stage_glb(hssd_root, stage_template, glb_cache=None):
    name = os.path.basename(stage_template)
    if glb_cache:
        cached = os.path.join(glb_cache, name + ".glb")
        if os.path.exists(cached):
            return cached
    cand = os.path.join(hssd_root, "stages", name + ".glb")
    return cand if os.path.exists(cand) else None


def import_glb(filepath):
    """Import a GLB and return the list of newly added mesh objects."""
    import bpy
    before = set(o.name for o in bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=filepath)
    after = [o for o in bpy.data.objects if o.name not in before]
    return after


def quat_wxyz_to_blender_matrix(translation, quat_wxyz, scale):
    """
    Map an HSSD object instance (Habitat Y-up frame) into a Blender world
    matrix (Z-up frame) via change-of-basis conjugation: T_blender = C @ T_hab @ C^-1.

    C is the Y-up->Z-up basis change (x,y,z)->(x,-z,y). bpy.ops.import_scene.gltf
    already applies C to each asset's local vertices, so the placement matrix
    must be conjugated by C (not merely left-multiplied) for rotation AND the
    non_uniform_scale axes to land correctly. Magnum quaternion order is [w,x,y,z].
    """
    import mathutils

    w, x, y, z = quat_wxyz
    sx, sy, sz = scale
    t_hab = (
        mathutils.Matrix.Translation(mathutils.Vector(translation))
        @ mathutils.Quaternion((w, x, y, z)).to_matrix().to_4x4()
        @ mathutils.Matrix.Diagonal((sx, sy, sz, 1.0))
    )
    c = mathutils.Matrix((
        (1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))
    return c @ t_hab @ c.inverted()


def assemble_scene(scene_instance_path, hssd_root, stage_only=False, glb_cache=None):
    import bpy

    with open(scene_instance_path) as f:
        scene = json.load(f)

    stage_tmpl = scene["stage_instance"]["template_name"]
    stage_glb = resolve_stage_glb(hssd_root, stage_tmpl, glb_cache)
    if stage_glb is None:
        raise FileNotFoundError(f"Stage GLB not found for {stage_tmpl}")
    print(f"[assemble] Loading stage: {stage_glb}")
    stage_objs = import_glb(stage_glb)
    print(f"[assemble] Stage added {len(stage_objs)} objects")

    if stage_only:
        print("[assemble] stage_only=True -> skipping object instances")
        return

    objs = scene.get("object_instances", [])
    print(f"[assemble] Placing {len(objs)} object instances...")
    placed, missing = 0, 0
    for idx, inst in enumerate(objs):
        tmpl = inst["template_name"]
        glb = resolve_object_glb(hssd_root, tmpl, glb_cache)
        if glb is None:
            missing += 1
            print(f"[assemble]   MISSING object glb: {tmpl}")
            continue
        new_objs = import_glb(glb)
        mat = quat_wxyz_to_blender_matrix(
            inst.get("translation", [0, 0, 0]),
            inst.get("rotation", [1, 0, 0, 0]),
            inst.get("non_uniform_scale", [1, 1, 1]),
        )
        for o in new_objs:
            if o.parent is None:
                o.matrix_world = mat @ o.matrix_world
        placed += 1
    print(f"[assemble] Placed {placed} objects, {missing} missing")


def fix_materials():
    import bpy
    fixed = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        if bsdf is None:
            continue
        base_color_link = bsdf.inputs["Base Color"].links[0] if bsdf.inputs["Base Color"].is_linked else None
        bsdf.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.3
        if base_color_link:
            from_sock = base_color_link.from_socket
            curve = nt.nodes.new("ShaderNodeRGBCurve")
            curve.mapping.extend = "EXTRAPOLATED"
            c = curve.mapping.curves[3]
            c.points[0].location = (0.0, 0.0)
            c.points.new(0.35, 0.20)
            c.points.new(0.70, 0.85)
            c.points[1].location = (1.0, 1.0)
            nt.links.new(from_sock, curve.inputs["Color"])
            nt.links.new(curve.outputs["Color"], bsdf.inputs["Emission Color"])
        else:
            bsdf.inputs["Emission Color"].default_value = tuple(bsdf.inputs["Base Color"].default_value)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.3
        fixed += 1
    print(f"[render] Fixed {fixed} materials (Metallic=0, Emission=0.3, curves)")


def setup_fisheye_camera(width, height, fov_rad):
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


def setup_lighting():
    """
    HSSD GLBs have real PBR textures but no baked lighting, so we provide
    bright, even illumination: a strong world ambient + a sun from above.
    """
    import bpy

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 0.5

    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun_data.energy = 15.0
    sun_data.angle = np.radians(15)
    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (np.radians(50), np.radians(35), 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene_instance", required=True,
                        help="Path to <scene>.scene_instance.json")
    parser.add_argument("--hssd_root", required=True,
                        help="Root of hssd-hab dataset")
    parser.add_argument("--traj_dir", required=True,
                        help="Dir with pre-extracted .npz trajectory files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov_deg", type=float, default=195.0)
    parser.add_argument("--depth_scale", type=float, default=10000.0)
    parser.add_argument("--max_depth_m", type=float, default=6.0)
    parser.add_argument("--stage_only", action="store_true",
                        help="Load only the stage GLB (fast coord-alignment test)")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="If >0, render only the first N episodes (testing)")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Render every Nth frame (testing)")
    parser.add_argument("--glb_cache", default=None,
                        help="Dir with pre-decompressed GLBs (KTX2→PNG)")
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fov_rad = np.radians(args.fov_deg)
    print(f"[render] Scene={args.scene}, FOV={args.fov_deg}deg, "
          f"res={args.width}x{args.height}, stage_only={args.stage_only}")

    bproc.init()
    import bpy

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = False

    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1.0

    assemble_scene(args.scene_instance, args.hssd_root,
                   stage_only=args.stage_only, glb_cache=args.glb_cache)

    fix_materials()
    setup_lighting()

    bpy.context.scene.use_nodes = True
    ctree = bpy.context.scene.node_tree
    ctree.nodes.clear()
    rl = ctree.nodes.new("CompositorNodeRLayers")
    bc = ctree.nodes.new("CompositorNodeBrightContrast")
    bc.inputs["Bright"].default_value = 0.05
    bc.inputs["Contrast"].default_value = 0.15
    comp = ctree.nodes.new("CompositorNodeComposite")
    ctree.links.new(rl.outputs["Image"], bc.inputs["Image"])
    ctree.links.new(bc.outputs["Image"], comp.inputs["Image"])

    setup_fisheye_camera(args.width, args.height, fov_rad)
    print(f"[render] Fisheye camera configured: equidistant, fov={args.fov_deg}deg")

    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    npz_files = sorted([f for f in os.listdir(args.traj_dir) if f.endswith(".npz")])
    if args.max_episodes > 0:
        npz_files = npz_files[:args.max_episodes]
    print(f"[render] {len(npz_files)} episode trajectory files")

    episode_info = []
    total = 0
    for ep_idx, nf in enumerate(npz_files):
        _, actions = load_trajectory(os.path.join(args.traj_dir, nf))
        idxs = list(range(0, actions.shape[0], args.frame_stride))
        episode_info.append((ep_idx, idxs))
        for i in idxs:
            bproc.camera.add_camera_pose(actions[i])
        total += len(idxs)
        print(f"[render] Episode {ep_idx}: {len(idxs)} frames (file={nf})")

    print(f"[render] Total frames: {total}")
    print("[render] Rendering...")
    data = bproc.renderer.render()
    print("[render] Render complete!")

    rgb_dir = os.path.join(args.output_dir, "observation.images.rgb")
    depth_dir = os.path.join(args.output_dir, "observation.images.depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rgb_images = data["colors"]
    depth_images = data["depth"]

    fi = 0
    for ep_idx, idxs in episode_info:
        for local_i, orig_i in enumerate(idxs):
            rgb = rgb_images[fi]
            Image.fromarray(rgb).save(
                os.path.join(rgb_dir, f"episode_{ep_idx:06d}_{orig_i:03d}.jpg"),
                quality=95)
            depth_m = np.clip(depth_images[fi], 0, args.max_depth_m)
            depth_u16 = (depth_m * args.depth_scale).astype(np.uint16)
            Image.fromarray(depth_u16, mode="I;16").save(
                os.path.join(depth_dir, f"episode_{ep_idx:06d}_{orig_i:03d}.png"))
            fi += 1
        print(f"[render] Episode {ep_idx}: saved {len(idxs)} frames")

    print(f"[render] Done! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
