"""
Pre-decompress KTX2/BasisU textures in HSSD object GLBs to PNG.

Blender 4.2's glTF addon cannot decode KHR_texture_basisu (KTX2). This script
uses gltf-transform's `ktxdecompress` CLI (which wraps Khronos KTX-Software)
to convert all unique object GLBs referenced by a scene_instance.json into
decompressed GLBs with PNG textures.

Usage:
    python3 decompress_glbs.py \
        --scene_instance /ssd5/datasets/hssd-hab/scenes/102344049.scene_instance.json \
        --hssd_root /ssd5/datasets/hssd-hab \
        --output_dir /tmp/opencode/hssd_fisheye_work/glbs_decompressed
"""
import argparse
import glob
import json
import os
import subprocess
import sys


def resolve_object_glb(hssd_root, template_name):
    cand = os.path.join(hssd_root, "objects", template_name[0], template_name + ".glb")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join(hssd_root, "objects", "**", template_name + ".glb"),
                     recursive=True)
    hits = [h for h in hits if "collider" not in os.path.basename(h)]
    return hits[0] if hits else None


def resolve_stage_glb(hssd_root, stage_template):
    name = os.path.basename(stage_template)
    cand = os.path.join(hssd_root, "stages", name + ".glb")
    return cand if os.path.exists(cand) else None


def needs_decompression(glb_path):
    try:
        import struct
        with open(glb_path, 'rb') as f:
            f.read(12)
            jlen = struct.unpack('<I', f.read(4))[0]
            f.read(4)
            j = json.loads(f.read(jlen))
        return 'KHR_texture_basisu' in (j.get('extensionsRequired') or [])
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_instance", required=True)
    parser.add_argument("--hssd_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.scene_instance) as f:
        scene = json.load(f)

    templates = set()
    stage_tmpl = scene["stage_instance"]["template_name"]
    templates.add(("stage", stage_tmpl))
    for inst in scene.get("object_instances", []):
        templates.add(("object", inst["template_name"]))

    env = os.environ.copy()
    env["PATH"] = "/home/rlan/.local/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/home/rlan/.local/bin:" + env.get("LD_LIBRARY_PATH", "")

    done, skipped, failed = 0, 0, 0
    for kind, tmpl in sorted(templates):
        if kind == "stage":
            src = resolve_stage_glb(args.hssd_root, tmpl)
        else:
            src = resolve_object_glb(args.hssd_root, tmpl)

        if src is None:
            print(f"  MISSING: {tmpl}")
            failed += 1
            continue

        name = os.path.basename(src)
        dst = os.path.join(args.output_dir, name)

        if os.path.exists(dst):
            skipped += 1
            continue

        if not needs_decompression(src):
            import shutil
            shutil.copy2(src, dst)
            skipped += 1
            continue

        print(f"  decompressing: {name} ...", end=" ", flush=True)
        try:
            result = subprocess.run(
                ["npx", "@gltf-transform/cli", "ktxdecompress", src, dst],
                env=env, timeout=60, capture_output=True, text=True,
            )
            if result.returncode == 0 and os.path.exists(dst):
                print("OK")
                done += 1
            else:
                print(f"FAILED: {result.stderr[:100]}")
                failed += 1
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\nDone: {done} decompressed, {skipped} copied/skipped, {failed} failed")
    print(f"Cache dir: {args.output_dir}")


if __name__ == "__main__":
    main()
