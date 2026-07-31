"""Forbidden-import smoke test for Phase 1 package-safe sage3d modules.

Runs in a fresh subprocess so preloaded pytest plugins cannot populate
``sys.modules``. Imports the modules that exist at this phase (issues #7 and
#8): frames, camera, episode_arrays, naming, io_ply, pointcloud,
publication, render_processing, and sage3d.cli._args. Asserts cv2, scipy,
trimesh, pxr, and isaacsim are absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MODULES = [
    "sage3d.frames",
    "sage3d.camera",
    "sage3d.episode_arrays",
    "sage3d.naming",
    "sage3d.io_ply",
    "sage3d.pointcloud",
    "sage3d.publication",
    "sage3d.render_processing",
    "sage3d.config",
    "sage3d.schemas",
    "sage3d.contract",
    "sage3d.cli._args",
]
_FORBIDDEN = ["cv2", "scipy", "trimesh", "pxr", "isaacsim"]

_PROBE = """
import sys
modules = {modules!r}
forbidden = {forbidden!r}
for m in modules:
    __import__(m)
loaded = set(sys.modules)
bad = set(forbidden) & loaded
assert not bad, f"forbidden modules loaded after import: {{bad}}"
print("OK")
""".format(modules=_MODULES, forbidden=_FORBIDDEN)


def test_phase1_package_safe_modules_import_no_forbidden_deps():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env={**__import__("os").environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"forbidden-import probe failed:\nstdout={proc.stdout}\n"
        f"stderr={proc.stderr}"
    )
    assert proc.stdout.strip() == "OK"