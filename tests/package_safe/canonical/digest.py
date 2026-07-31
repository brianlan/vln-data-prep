"""Canonical digest primitives for SAGE3D Phase 0b evidence.

Domain-separated, schema-versioned SHA-256 framing per SAGE3D_REFACTOR_PLAN.md
revision 8 (Canonical digest framing, binding). Every variable-length name,
metadata block, and payload is preceded by an unsigned 64-bit big-endian byte
length; collections begin with an unsigned 64-bit big-endian item count.

Package-safe: stdlib (``hashlib``, ``struct``, ``json``, ``pathlib``) only.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

DOMAIN_TAG = "sage3d-digest-v1"
DIGEST_KINDS = ("trajectory", "rendered_root", "packaged_root", "evidence")
_SCHEMA_VERSION = 1
_U64 = struct.Struct(">Q")


# --- low-level framing helpers ----------------------------------------------

def _u64(value: int) -> bytes:
    return _U64.pack(value)


def _frame_bytes(data: bytes) -> bytes:
    """Length-prefix raw bytes with a u64 big-endian byte count."""
    return _u64(len(data)) + data


def _frame_text(text: str) -> bytes:
    """Length-prefix UTF-8 text with a u64 big-endian byte count."""
    return _frame_bytes(text.encode("utf-8"))


class FramingWriter:
    """Streaming SHA-256 over a domain-separated, schema-versioned byte stream.

    Use as a context manager; call ``update`` with framed bytes, then read
    ``hexdigest`` on close.
    """

    def __init__(self, digest_kind: str) -> None:
        if digest_kind not in DIGEST_KINDS:
            raise ValueError(
                f"invalid digest kind {digest_kind!r}; expected one of {DIGEST_KINDS}"
            )
        self._hasher = hashlib.sha256()
        self._hasher.update(_frame_text(f"{DOMAIN_TAG}:{digest_kind}"))
        self._hasher.update(_u64(_SCHEMA_VERSION))

    def update(self, data: bytes) -> "FramingWriter":
        self._hasher.update(data)
        return self

    @property
    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


# --- canonical JSON ----------------------------------------------------------

def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON per the plan: sort_keys, compact separators, no NaN."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_json(digest_kind: str, value: Any) -> str:
    """SHA-256 of a canonical-JSON value under the given digest kind."""
    payload = canonical_json_bytes(value)
    w = FramingWriter(digest_kind)
    w.update(_frame_bytes(payload))
    return w.hexdigest


# --- array framing (NumPy) ---------------------------------------------------

def _frame_array(name: str, key: str, array: Any) -> bytes:
    """Frame a single NumPy array: name, key, dtype.str, rank, shape, order, bytes."""
    import numpy as np  # local import: only needed for array digests

    # ponytail: numpy import inside the function so the module stays importable
    # in a numpy-free context for pure-JSON digests.
    if not isinstance(array, np.ndarray):
        raise TypeError(f"expected ndarray for {name}/{key}, got {type(array)}")
    dtype_str = str(array.dtype)
    # Ensure C-order contiguous bytes for stable hashing.
    c_bytes = np.ascontiguousarray(array).tobytes(order="C")
    shape = array.shape
    out = b""
    out += _frame_text(name)
    out += _frame_text(key)
    out += _frame_text(dtype_str)
    out += _u64(array.ndim)  # rank
    for dim in shape:
        out += _u64(dim)
    out += _frame_text("C")  # literal storage order
    out += _u64(len(c_bytes))
    out += c_bytes
    return out


def digest_arrays(digest_kind: str, name: str, arrays: dict[str, Any]) -> str:
    """SHA-256 over a framed, ordered collection of named NumPy arrays."""
    w = FramingWriter(digest_kind)
    w.update(_frame_text(name))
    w.update(_u64(len(arrays)))
    for key in arrays:
        w.update(_frame_array(name, key, arrays[key]))
    return w.hexdigest


# --- tree / directory hashing ------------------------------------------------

def _normalize_relative_path(rel: str) -> str:
    """Normalize to POSIX, reject absolute/parent-traversal paths."""
    if rel.startswith("/"):
        raise ValueError(f"absolute path rejected: {rel}")
    if ".." in rel.split("/"):
        raise ValueError(f"parent-traversal path rejected: {rel}")
    posix = rel.replace(os.sep, "/")
    if posix in ("", ".", ".."):
        raise ValueError(f"invalid normalized path: {rel!r}")
    return posix


def digest_file(digest_kind: str, path: Path) -> str:
    """SHA-256 of a regular file's exact bytes (PLY contract)."""
    if path.is_symlink():
        raise ValueError(f"symlink rejected: {path}")
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    h = hashlib.sha256()
    h.update(_frame_text(f"{DOMAIN_TAG}:{digest_kind}"))
    h.update(_u64(_SCHEMA_VERSION))
    h.update(_frame_text(path.name))
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_directory(digest_kind: str, root: Path) -> str:
    """SHA-256 over a directory tree, ordered by normalized relative POSIX path.

    Rejects symlinks, non-regular/non-directory entries, absolute/parent
    traversal, and duplicate normalized names.
    """

    def _walk(current: Path, base: str):
        entries = sorted(os.listdir(current))
        seen: dict[str, str] = {}
        for entry in entries:
            full = current / entry
            rel = f"{base}/{entry}" if base else entry
            norm = _normalize_relative_path(rel)
            if norm in seen:
                raise ValueError(f"duplicate normalized name: {norm}")
            seen[norm] = norm
            if full.is_symlink():
                raise ValueError(f"symlink rejected: {full}")
            if full.is_dir():
                yield (norm, "dir", None)
                yield from _walk(full, norm)
            elif full.is_file():
                yield (norm, "file", full)
            else:
                raise ValueError(f"non-regular/non-directory entry: {full}")

    w = FramingWriter(digest_kind)
    w.update(_frame_text(str(root.name)))
    components = list(_walk(root, ""))
    w.update(_u64(len(components)))
    for norm, kind, path in components:
        w.update(_frame_text(norm))
        w.update(_frame_text(kind))
        if kind == "file":
            h = hashlib.sha256()
            with path.open("rb") as f:  # type: ignore[union-attr]
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            w.update(_frame_text(h.hexdigest()))
    return w.hexdigest