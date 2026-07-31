"""Pytest configuration for package-safe tests.

These tests run under ``$SAGE3D_PACKAGE_PYTHON`` and must not import Isaac-side
modules (``cv2``, ``scipy``, ``trimesh``, ``pxr``, ``isaacsim``). They import only
the repository-root modules that are package-safe at the current phase. The
repository is run from its checkout, so the repo root is prepended to
``sys.path`` here instead of depending on an installation step.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make the package_safe test helpers (fixtures) importable as top-level
# modules. artifact_parsers has been promoted to sage3d_canonical/parsers.py
# at the repo root (issue #36).
_PACKAGE_SAFE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_SAFE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SAFE_DIR))