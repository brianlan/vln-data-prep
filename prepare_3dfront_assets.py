"""Prepare the assets needed to render one 3D-FRONT scene.

The official download is distributed as one scene ZIP, one architecture-texture
ZIP, and four 3D-FUTURE model ZIPs. BlenderProc expects those files in extracted
directories. This utility extracts only the JSON, furniture models, and
architecture textures referenced by one scene into a persistent per-scene
cache.

Recent 3D-FRONT JSON files often identify an architecture texture through the
material ``jid`` while leaving the legacy ``texture`` URL empty. BlenderProc
2.8 still looks at the URL field, so this utility writes a normalized scene
JSON whose URL points at the matching locally downloaded texture UUID.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile


SCENE_ZIP = "3D-FRONT.zip"
TEXTURE_ZIP = "3D-FRONT-texture.zip"
MODEL_ZIPS = tuple(f"3D-FUTURE-model-part{part}.zip" for part in range(1, 5))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def validate_scene_id(scene):
    if not scene or any(char not in "0123456789abcdef-" for char in scene.lower()):
        raise ValueError(f"Invalid 3D-FRONT scene ID: {scene!r}")


def require_archives(dataset_root):
    paths = [dataset_root / SCENE_ZIP, dataset_root / TEXTURE_ZIP]
    paths.extend(dataset_root / name for name in MODEL_ZIPS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing 3D-FRONT archive(s):\n  " + "\n  ".join(missing))
    return paths


def read_scene_json(scene_zip_path, scene):
    member = f"3D-FRONT/{scene}.json"
    with zipfile.ZipFile(scene_zip_path) as archive:
        try:
            raw = archive.read(member)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Scene {scene} is not present in {scene_zip_path}"
            ) from exc
    return json.loads(raw)


def used_furniture(data):
    refs = {
        child.get("ref")
        for room in data.get("scene", {}).get("room", [])
        for child in room.get("children", [])
        if "furniture" in child.get("instanceid", "")
    }
    furniture = [item for item in data.get("furniture", []) if item.get("uid") in refs]
    jids = {item["jid"] for item in furniture if item.get("jid")}
    return furniture, jids


def archive_directory_index(archive, root_name):
    index = {}
    for member in archive.namelist():
        parts = PurePosixPath(member).parts
        if len(parts) >= 3 and parts[0] == root_name:
            index.setdefault(parts[1], []).append(member)
    return index


def safe_extract_members(archive, members, source_prefix_parts, destination):
    extracted = 0
    for member in members:
        info = archive.getinfo(member)
        if info.is_dir():
            continue
        parts = PurePosixPath(member).parts
        relative_parts = parts[source_prefix_parts:]
        if not relative_parts or any(part in ("", ".", "..") for part in relative_parts):
            raise RuntimeError(f"Unsafe archive member: {member}")
        target = destination.joinpath(*relative_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        extracted += 1
    return extracted


def extract_furniture(dataset_root, model_root, required_jids):
    remaining = set(required_jids)
    extracted_models = set()
    extracted_files = 0

    for zip_name in MODEL_ZIPS:
        zip_path = dataset_root / zip_name
        archive_root = zip_name.removesuffix(".zip")
        with zipfile.ZipFile(zip_path) as archive:
            index = archive_directory_index(archive, archive_root)
            for jid in sorted(remaining & index.keys()):
                destination = model_root / jid
                extracted_files += safe_extract_members(
                    archive,
                    index[jid],
                    source_prefix_parts=2,
                    destination=destination,
                )
                if not (destination / "raw_model.obj").is_file():
                    raise RuntimeError(
                        f"Extracted furniture {jid} has no raw_model.obj in {zip_path}"
                    )
                extracted_models.add(jid)
            remaining -= extracted_models

    return extracted_models, remaining, extracted_files


def texture_id_from_url(url):
    if not url:
        return None
    parts = url.rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else None


def normalize_material_textures(data, available_texture_ids):
    required_ids = set()
    normalized_count = 0
    for material in data.get("material", []):
        for field in ("texture", "normaltexture"):
            texture_id = texture_id_from_url(material.get(field, ""))
            if texture_id in available_texture_ids:
                required_ids.add(texture_id)

        if not material.get("texture") and not material.get("useColor", False):
            texture_id = material.get("jid")
            if texture_id in available_texture_ids:
                # Only the parent directory name is consumed by BlenderProc.
                material["texture"] = (
                    f"https://local.3dfront.invalid/{texture_id}/texture.png"
                )
                required_ids.add(texture_id)
                normalized_count += 1
    return required_ids, normalized_count


def extract_architecture_textures(texture_zip_path, texture_root, data):
    archive_root = TEXTURE_ZIP.removesuffix(".zip")
    with zipfile.ZipFile(texture_zip_path) as archive:
        index = archive_directory_index(archive, archive_root)
        required_ids, normalized_count = normalize_material_textures(data, set(index))
        extracted_files = 0
        for texture_id in sorted(required_ids):
            extracted_files += safe_extract_members(
                archive,
                index[texture_id],
                source_prefix_parts=2,
                destination=texture_root / texture_id,
            )
    return required_ids, normalized_count, extracted_files


def main():
    args = parse_args()
    validate_scene_id(args.scene)
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    require_archives(dataset_root)

    scene_root = output_root / args.scene
    model_root = scene_root / "3D-FUTURE-model"
    texture_root = scene_root / "3D-FRONT-texture"
    scene_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    texture_root.mkdir(parents=True, exist_ok=True)

    data = read_scene_json(dataset_root / SCENE_ZIP, args.scene)
    furniture, required_jids = used_furniture(data)
    # The official loader loads every furniture record before placing used
    # instances. Dropping unused records makes per-scene extraction sufficient.
    data["furniture"] = furniture

    texture_ids, normalized_count, texture_files = extract_architecture_textures(
        dataset_root / TEXTURE_ZIP, texture_root, data
    )
    extracted_models, missing_models, model_files = extract_furniture(
        dataset_root, model_root, required_jids
    )

    normalized_json = scene_root / f"{args.scene}.json"
    with normalized_json.open("w", encoding="utf-8") as output:
        json.dump(data, output)

    manifest = {
        "scene": args.scene,
        "scene_json": str(normalized_json),
        "used_furniture_records": len(furniture),
        "required_furniture_models": len(required_jids),
        "extracted_furniture_models": len(extracted_models),
        "missing_furniture_models": sorted(missing_models),
        "architecture_textures": len(texture_ids),
        "normalized_material_texture_references": normalized_count,
        "extracted_model_files": model_files,
        "extracted_texture_files": texture_files,
    }
    with (scene_root / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)

    print(
        f"[assets] Scene {args.scene}: {len(extracted_models)}/{len(required_jids)} "
        f"furniture models, {len(texture_ids)} architecture textures"
    )
    if missing_models:
        print(
            "[assets] WARNING: public 3D-FUTURE archives do not contain "
            f"{len(missing_models)} referenced model(s): "
            + ", ".join(sorted(missing_models))
        )
    print(f"[assets] Normalized scene JSON: {normalized_json}")


if __name__ == "__main__":
    main()
