"""Binary little-endian PLY writer + metadata reader (numpy only).

``write_binary_pointcloud`` reproduces the exact byte stream produced by
``generate_sage3d_trajectories.py``: ASCII header, ``float32`` x/y/z, ``uint8``
r/g/b (fixed 160/160/160), ``<fffBBB`` struct record.

``read_binary_pointcloud_metadata`` parses only the header so package-safe
validators can read vertex counts and bounds without loading geometry into a
non-package lane.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_PLY_HEADER_LINES = (
    "ply\n",
    "format binary_little_endian 1.0\n",
    "comment SAGE3D collision mesh voxel point cloud\n",
    "element vertex {vertex_count}\n",
    "property float x\n",
    "property float y\n",
    "property float z\n",
    "property uchar red\n",
    "property uchar green\n",
    "property uchar blue\n",
    "end_header\n",
)
_PLY_RECORD = struct.Struct("<fffBBB")
_PLY_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def write_binary_pointcloud(path: Path, points: np.ndarray) -> None:
    """Write a PLY pointcloud with the exact legacy byte layout.

    ``points`` is an ``(N, 3)`` array; colors are fixed at ``(160, 160, 160)``
    to match the legacy writer. Output bytes are byte-for-byte stable given the
    same point order and dtypes.
    """
    points = np.asarray(points, dtype=np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment SAGE3D collision mesh voxel point cloud\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        for x, y, z in points:
            file.write(_PLY_RECORD.pack(float(x), float(y), float(z), 160, 160, 160))


def read_binary_pointcloud_metadata(path: Path) -> dict:
    """Parse the PLY header and return vertex count + property layout.

    Returns a dict with ``vertex_count`` (int) and ``properties`` (list[str]).
    Raises ``ValueError`` for unsupported formats or missing vertex elements.
    """
    vertex_count: int | None = None
    properties: list[str] = []
    with path.open("rb") as file:
        magic = file.readline()
        if magic.strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        format_line = file.readline().decode("ascii").strip()
        if format_line != "format binary_little_endian 1.0":
            raise ValueError(f"{path} has unsupported PLY format: {format_line!r}")
        for raw in file:
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("element vertex "):
                vertex_count = int(line.rsplit(" ", 1)[1])
            elif line.startswith("property "):
                properties.append(line.split(" ", 1)[1])
    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element")
    return {"vertex_count": vertex_count, "properties": properties}