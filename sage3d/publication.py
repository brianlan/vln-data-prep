"""Non-destructive publication primitives (stdlib + pathlib only).

All three SAGE3D producers (generate, render, package) publish atomically by
writing into a sibling staging directory and renaming it onto an absent
final target. This module owns the four safety primitives they share:

1. :func:`assert_target_absent` — refuse any existing target (regular file,
   directory, symlink, dangling symlink, special entry).
2. :func:`create_staging_directory` — allocate a real sibling staging
   directory via ``tempfile.mkdtemp`` under the resolved final-target parent.
3. :func:`assert_staging_entries_regular` — walk staging with ``lstat`` and
   reject symlinked directories/files and FIFO/socket/device entries before
   publication.
4. :func:`atomic_publish_directory` — recheck target absence and rename the
   staging directory onto the target on the same filesystem, with no copy
   fallback.

The contract is *cooperative single-publisher*: the documented race after the
final absence recheck is outside this module's guarantees. Cross-filesystem
copy fallback is deliberately forbidden because it would weaken atomicity.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def validate_real_directory(path: Path) -> None:
    """Raise if ``path`` is absent, a symlink, or not a directory by lstat."""
    if not os.path.lexists(path):
        raise FileNotFoundError(f"expected existing real directory: {path}")
    if os.path.islink(path):
        raise ValueError(f"refusing symlinked directory: {path}")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"not a directory: {path}")


def assert_target_absent(target: Path) -> None:
    """Refuse any existing ``target`` (file/dir/symlink/dangling/special)."""
    if os.path.lexists(target):
        raise FileExistsError(f"publication target already exists: {target}")


def _require_same_device(a: Path, b: Path) -> None:
    """Ensure two paths live on the same filesystem (rename-safe)."""
    a_stat = os.lstat(a)
    b_stat = os.lstat(b)
    if a_stat.st_dev != b_stat.st_dev:
        raise OSError(
            f"staging and target are on different filesystems: "
            f"{a} (dev={a_stat.st_dev}) vs {b} (dev={b_stat.st_dev})"
        )


def create_staging_directory(final_target: Path, prefix: str) -> Path:
    """Allocate a real sibling staging directory for ``final_target``.

    The staging directory is created with ``tempfile.mkdtemp`` under the
    *resolved* parent of ``final_target`` so the final ``os.rename`` is on one
    filesystem. The parent must already be a real directory (not a symlink);
    the staging directory is verified with ``lstat`` and confirmed on the same
    device as its parent.
    """
    final_target = Path(final_target)
    if not prefix:
        raise ValueError("staging prefix must be non-empty")
    parent = final_target.parent.resolve(strict=True)
    validate_real_directory(parent)
    staging = Path(
        tempfile.mkdtemp(prefix=prefix, dir=str(parent))
    )
    validate_real_directory(staging)
    _require_same_device(staging, parent)
    return staging


def create_named_directory(parent: Path, name: str) -> Path:
    """Create one absent, non-symlinked child on the parent's filesystem.

    Used by the canonical harness for its evidence/run directory tree. The
    parent must already be a real directory; ``name`` must be a single path
    component (not ``.``, ``..``, or contain ``/``). The child must not exist by
    ``lexists``. All safety checks (lstat real parent, ``lexists`` refusal,
    same-device verification) delegate to the same primitives as
    :func:`create_staging_directory` so no independent allocation formula
    survives.
    """
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"invalid directory name: {name!r}")
    parent = Path(parent)
    validate_real_directory(parent)
    target = parent / name
    assert_target_absent(target)
    target.mkdir()
    validate_real_directory(target)
    _require_same_device(target, parent)
    return target


def assert_staging_entries_regular(staging: Path) -> None:
    """Walk ``staging`` with ``lstat`` and reject symlinks/special entries.

    Regular files and directories are accepted. Symlinks, FIFOs, sockets, and
    device files anywhere under ``staging`` are refused so publication never
    renames an unsafe tree onto the final target.
    """
    staging = Path(staging)
    validate_real_directory(staging)
    for entry in staging.rglob("*"):
        info = os.lstat(entry)
        mode = info.st_mode
        if os.path.islink(entry):
            raise ValueError(f"refusing symlink in staging: {entry}")
        if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
            raise ValueError(f"refusing FIFO/socket in staging: {entry}")
        if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
            raise ValueError(f"refusing device file in staging: {entry}")
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"refusing unexpected entry type in staging: {entry}")


def atomic_publish_directory(staging: Path, target: Path) -> Path:
    """Atomically rename ``staging`` onto the absent ``target``.

    Pre-conditions enforced here:

    * ``staging`` is a real directory (lstat-verified).
    * staging entries are all regular files/directories (no symlinks/special).
    * ``target`` is absent immediately before the rename (rechecked here).
    * staging and target parent are on the same filesystem.

    Cross-filesystem copy fallback is intentionally *not* provided; a device
    mismatch raises so the caller can fix allocation. The documented race after
    the final absence recheck (a non-cooperating process creating ``target``
    between the check and the rename) is outside the cooperative
    single-publisher contract.
    """
    staging = Path(staging)
    target = Path(target)
    validate_real_directory(staging)
    assert_staging_entries_regular(staging)
    parent = target.parent.resolve(strict=True)
    validate_real_directory(parent)
    _require_same_device(staging, parent)
    # Recheck absence immediately before rename; a cooperative publisher never
    # creates the target, so a present target here is an error.
    assert_target_absent(target)
    os.rename(staging, target)
    return target