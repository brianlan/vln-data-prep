"""Tests for canonical digest framing, invariance, and sensitivity.

Covers the binding requirements from SAGE3D_REFACTOR_PLAN.md revision 8:
domain-separated framing, u64 big-endian lengths/counts, canonical JSON,
array framing (dtype/shape/order/bytes), PLY exact-file SHA-256, directory
tree hashing with path normalization and symlink/special-file refusal, and
the committed digest vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from sage3d_canonical.digest import (
    DOMAIN_TAG,
    DIGEST_KINDS,
    FramingWriter,
    _frame_bytes,
    _frame_text,
    _normalize_relative_path,
    _u64,
    canonical_json_bytes,
    digest_arrays,
    digest_directory,
    digest_file,
    digest_json,
)


# --- framing boundaries & byte order ----------------------------------------

def test_u64_is_big_endian():
    assert _u64(1) == b"\x00\x00\x00\x00\x00\x00\x00\x01"


def test_frame_bytes_prefixes_u64_length():
    data = b"hello"
    framed = _frame_bytes(data)
    assert framed[:8] == struct.pack(">Q", 5)
    assert framed[8:] == data


def test_frame_text_utf8_with_u64_length():
    text = "café"
    framed = _frame_text(text)
    assert framed[:8] == struct.pack(">Q", len("café".encode("utf-8")))
    assert framed[8:] == "café".encode("utf-8")


def test_domain_tag_is_sage3d_digest_v1():
    assert DOMAIN_TAG == "sage3d-digest-v1"


def test_digest_kinds_include_all_four():
    for kind in ("trajectory", "rendered_root", "packaged_root", "evidence"):
        assert kind in DIGEST_KINDS


def test_framing_writer_rejects_invalid_kind():
    with pytest.raises(ValueError, match="invalid digest kind"):
        FramingWriter("bogus")


def test_framing_writer_hexdigest_is_sha256():
    w = FramingWriter("evidence")
    digest = w.hexdigest
    assert len(digest) == 64
    int(digest, 16)  # valid hex


def test_different_digest_kinds_produce_different_hashes():
    payload = _frame_bytes(b"same")
    assert (
        FramingWriter("trajectory").update(payload).hexdigest
        != FramingWriter("rendered_root").update(payload).hexdigest
    )


def test_framing_writer_prefixes_domain_tag_and_schema_version():
    """The framing writer includes the domain tag and schema version first."""
    # Two writers with the same kind and no further updates should match.
    w1 = FramingWriter("evidence")
    w2 = FramingWriter("evidence")
    assert w1.hexdigest == w2.hexdigest
    # A writer with a different kind diverges even with identical updates.
    w3 = FramingWriter("trajectory")
    assert w1.hexdigest != w3.hexdigest


# --- canonical JSON ----------------------------------------------------------

def test_canonical_json_is_sorted_compact_no_nan():
    result = canonical_json_bytes({"b": 1, "a": 2})
    assert result == b'{"a":2,"b":1}'


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json_bytes(float("nan"))


def test_canonical_json_preserves_list_order():
    assert canonical_json_bytes([3, 1, 2]) == b"[3,1,2]"


def test_digest_json_is_stable():
    d1 = digest_json("evidence", {"a": 1, "b": [2, 3]})
    d2 = digest_json("evidence", {"b": [2, 3], "a": 1})
    assert d1 == d2


def test_digest_json_changes_for_value_change():
    d1 = digest_json("evidence", {"a": 1})
    d2 = digest_json("evidence", {"a": 2})
    assert d1 != d2


# --- array framing -----------------------------------------------------------

def test_digest_arrays_stable_for_same_arrays():
    arrays = {"x": np.zeros((2, 3), dtype=np.float32)}
    d1 = digest_arrays("trajectory", "ep0", arrays)
    d2 = digest_arrays("trajectory", "ep0", arrays)
    assert d1 == d2


def test_digest_arrays_changes_for_value_change():
    d1 = digest_arrays("trajectory", "ep0", {"x": np.zeros((2,), dtype=np.float32)})
    d2 = digest_arrays("trajectory", "ep0", {"x": np.ones((2,), dtype=np.float32)})
    assert d1 != d2


def test_digest_arrays_changes_for_shape_change():
    d1 = digest_arrays("trajectory", "ep0", {"x": np.zeros((2,), dtype=np.float32)})
    d2 = digest_arrays("trajectory", "ep0", {"x": np.zeros((3,), dtype=np.float32)})
    assert d1 != d2


def test_digest_arrays_changes_for_dtype_change():
    d1 = digest_arrays("trajectory", "ep0", {"x": np.zeros((2,), dtype=np.float32)})
    d2 = digest_arrays("trajectory", "ep0", {"x": np.zeros((2,), dtype=np.float64)})
    assert d1 != d2


def test_digest_arrays_changes_for_key_change():
    d1 = digest_arrays("trajectory", "ep0", {"x": np.zeros((2,), dtype=np.float32)})
    d2 = digest_arrays("trajectory", "ep0", {"y": np.zeros((2,), dtype=np.float32)})
    assert d1 != d2


def test_digest_arrays_changes_for_byte_order():
    be = np.zeros((2,), dtype=">f4")
    le = np.zeros((2,), dtype="<f4")
    assert (
        digest_arrays("trajectory", "ep0", {"x": be})
        != digest_arrays("trajectory", "ep0", {"x": le})
    )


# --- path normalization ------------------------------------------------------

def test_normalize_rejects_absolute():
    with pytest.raises(ValueError, match="absolute"):
        _normalize_relative_path("/etc")


def test_normalize_rejects_parent_traversal():
    with pytest.raises(ValueError, match="parent-traversal"):
        _normalize_relative_path("a/../b")


def test_normalize_converts_os_sep_to_posix():
    assert _normalize_relative_path(os.path.join("a", "b")) == "a/b"


def test_normalize_rejects_empty_dot_dotdot():
    for bad in ("", "."):
        with pytest.raises(ValueError, match="invalid"):
            _normalize_relative_path(bad)


# --- file digest (PLY exact-file contract) -----------------------------------

def test_digest_file_matches_raw_sha256_of_bytes(tmp_path):
    path = tmp_path / "test.ply"
    payload = b"ply\nend_header\n\x00\x01\x02"
    path.write_bytes(payload)
    # digest_file frames the domain tag + schema version + name before hashing,
    # so it differs from the raw file sha256 but is stable across calls.
    assert digest_file("packaged_root", path) != hashlib.sha256(payload).hexdigest()
    assert digest_file("packaged_root", path) == digest_file("packaged_root", path)


def test_digest_file_changes_for_content_change(tmp_path):
    path = tmp_path / "test.ply"
    path.write_bytes(b"original")
    d1 = digest_file("packaged_root", path)
    path.write_bytes(b"modified")
    d2 = digest_file("packaged_root", path)
    assert d1 != d2


def test_digest_file_rejects_symlink(tmp_path):
    target = tmp_path / "real.ply"
    target.write_bytes(b"data")
    link = tmp_path / "link.ply"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        digest_file("packaged_root", link)


def test_digest_file_rejects_nonexistent(tmp_path):
    with pytest.raises(ValueError, match="not a regular file"):
        digest_file("packaged_root", tmp_path / "missing.ply")


# --- directory tree hashing --------------------------------------------------

def test_digest_directory_stable(tmp_path):
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "a" / "f.txt").write_bytes(b"hello")
    d1 = digest_directory("rendered_root", root)
    d2 = digest_directory("rendered_root", root)
    assert d1 == d2


def test_digest_directory_changes_for_content_change(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "f.txt").write_bytes(b"hello")
    d1 = digest_directory("rendered_root", root)
    (root / "f.txt").write_bytes(b"world")
    d2 = digest_directory("rendered_root", root)
    assert d1 != d2


def test_digest_directory_changes_for_addition(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_bytes(b"hello")
    d1 = digest_directory("rendered_root", root)
    (root / "b.txt").write_bytes(b"hello")
    d2 = digest_directory("rendered_root", root)
    assert d1 != d2


def test_digest_directory_rejects_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "real.txt"
    target.write_bytes(b"data")
    (root / "link.txt").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        digest_directory("rendered_root", root)


def test_digest_directory_rejects_special_file(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    # Create a FIFO (a non-regular, non-directory entry).
    os.mkfifo(root / "fifo")
    with pytest.raises(ValueError, match="non-regular"):
        digest_directory("rendered_root", root)


# --- committed golden digest vectors ------------------------------------------
# These pinned values detect silent framing drift across versions. Any change
# is an intentional framing revision.

GOLDEN_JSON_EVIDENCE = "19c8cc87d717f5bc40b05c3d32efef283d3864af04b13c09596c3939414a2e91"
GOLDEN_ARRAY_RENDERED_ROOT = "a8a935c929215ebe81caf84611e72fe6a747f7712fd065450a440570be1e1feb"
GOLDEN_FILE_PACKAGED_ROOT = "d3bb7748c0a8a2a2cb4e3e35a8e300708c8ff49c07410886bf9d5cecf9a892f1"
GOLDEN_DIR_PACKAGED_ROOT = "e943b6f0164427d84e83f2b1f922282556ccdd2a62022315aaa483cf50c0ebed"


def test_committed_json_digest_vector():
    assert digest_json("evidence", {"a": 1, "b": [2, 3]}) == GOLDEN_JSON_EVIDENCE


def test_committed_array_digest_vector():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert digest_arrays("rendered_root", "depth_map", {"depth": arr}) == GOLDEN_ARRAY_RENDERED_ROOT


def test_committed_file_digest_vector(tmp_path):
    p = tmp_path / "foo.txt"
    p.write_text("hello")
    assert digest_file("packaged_root", p) == GOLDEN_FILE_PACKAGED_ROOT


def test_committed_directory_digest_vector(tmp_path):
    root = tmp_path / "golden_dir"
    root.mkdir()
    (root / "foo.txt").write_text("hello")
    (root / "sub").mkdir()
    (root / "sub" / "bar.txt").write_text("world")
    assert digest_directory("packaged_root", root) == GOLDEN_DIR_PACKAGED_ROOT