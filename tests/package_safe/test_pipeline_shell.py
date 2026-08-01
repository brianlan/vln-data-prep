"""Tests for the Phase 6 pipeline shell (issue #28).

Covers the destructive-path guard matrix, guarded disposable WORK_ROOT
creation, operator-provisioned OUTPUT_ROOT, source-subtree refusal, the
allocator -> render (rgb/depth) -> finalizer -> package -> validate flow, and
force/plan-only behavior. Interpreters are stubbed so the whole orchestration
runs without Isaac or GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHELL = REPO_ROOT / "run_pipeline_sage3d.sh"

STUB = r"""#!/usr/bin/env python3
# Stub interpreter for pipeline shell tests. Mimics the module CLIs just
# enough for the shell flow (directories + printed paths) without Isaac/GPU.
import os
import sys
from pathlib import Path


def _log(line: str) -> None:
    log_path = Path(os.environ["SAGE3D_SHELL_STUB_LOG"])
    with log_path.open("a") as f:
        f.write(line + "\n")


def _flag(args: list[str], name: str):
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            return args[i + 1]
    return None


def main() -> int:
    argv = sys.argv[1:]
    _log(" ".join(argv))
    if not argv:
        return 0
    if argv[0] == "-m":
        module = argv[1]
        args = argv[2:]
    else:
        module = Path(argv[0]).name
        args = argv[1:]

    if module == "sage3d.cli.generate":
        out = Path(_flag(args, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "trajectory_manifest.json").write_text('{"episodes": []}')
        (out / "episode_000000.npz").write_bytes(b"stub")
        (out / "pointcloud.ply").write_bytes(b"stub")
    elif module == "sage3d.cli.create_staging":
        target = Path(_flag(args, "--final-target"))
        staging = target.parent / (".rendered." + target.name)
        staging.mkdir(parents=True, exist_ok=True)
        print(staging.resolve())
    elif module == "sage3d.cli.render":
        staging = Path(_flag(args, "--staging-root"))
        mode = _flag(args, "--mode")
        (staging / f"observation.images.{mode}").mkdir(parents=True, exist_ok=True)
    elif module == "sage3d.cli.finalize_render":
        out = Path(_flag(args, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
    elif module == "sage3d.cli.package":
        out = Path(_flag(args, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
    elif module == "check_package.py":
        print("[check_package:validate] ELIGIBLE")
    else:
        print(f"unknown module: {module}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


@pytest.fixture
def shell_env(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """Write the stub interpreter and return (tmp, env, stub_log)."""
    stub = tmp_path / "stub_python.py"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / "stub.log"
    env = os.environ.copy()
    env["SAGE3D_ISAAC_PYTHON"] = str(stub)
    env["SAGE3D_PACKAGE_PYTHON"] = str(stub)
    env["SAGE3D_SHELL_STUB_LOG"] = str(log)
    return tmp_path, env, log


def run_shell(
    args: list[str],
    env: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SHELL)] + args,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def read_log(log: Path) -> list[str]:
    if not log.exists():
        return []
    return log.read_text().splitlines()


# --- help and argument parsing ----------------------------------------------


def test_help_exits_two(shell_env, tmp_path):
    """A bare --help is caught by the pre-argument guard and prints usage with
    exit code 2, matching the legacy shell behavior."""
    tmp, env, _ = shell_env
    result = run_shell(["--help"], env, tmp_path)
    assert result.returncode == 2
    assert "Usage: bash run_pipeline_sage3d.sh" in result.stdout


def test_no_scene_exits_two(shell_env):
    tmp, env, _ = shell_env
    result = run_shell([], env, tmp)
    assert result.returncode == 2


def test_unknown_arg_exits_two(shell_env):
    tmp, env, _ = shell_env
    result = run_shell(["839920", "--bogus"], env, tmp)
    assert result.returncode == 2


# --- path-guard matrix ------------------------------------------------------


def test_non_numeric_scene_refused(shell_env):
    tmp, env, log = shell_env
    result = run_shell(["abc"], env, tmp)
    assert result.returncode != 0
    assert "scene ID must be numeric" in result.stderr
    assert read_log(log) == []  # no interpreter was invoked


def test_missing_output_root_refused(shell_env):
    tmp, env, log = shell_env
    result = run_shell(["839920", "--output-root", str(tmp / "missing")], env, tmp)
    assert result.returncode != 0
    assert "OUTPUT_ROOT must already exist" in result.stderr
    assert read_log(log) == []


def test_file_output_root_refused(shell_env):
    tmp, env, log = shell_env
    a_file = tmp / "afile"
    a_file.write_text("x")
    result = run_shell(["839920", "--output-root", str(a_file)], env, tmp)
    assert result.returncode != 0
    assert "real directory" in result.stderr
    assert read_log(log) == []


def test_symlinked_output_root_refused(shell_env):
    tmp, env, log = shell_env
    real = tmp / "real"
    real.mkdir()
    link = tmp / "link"
    link.symlink_to(real, target_is_directory=True)
    result = run_shell(["839920", "--output-root", str(link)], env, tmp)
    assert result.returncode != 0
    assert "must not be a symlink" in result.stderr
    assert read_log(log) == []


def test_work_root_parent_missing_refused(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(tmp / "no" / "child")],
        env,
        tmp,
    )
    assert result.returncode != 0
    assert "WORK_ROOT parent" in result.stderr
    assert read_log(log) == []


def test_work_root_dotdot_basename_refused(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(tmp / "..")],
        env,
        tmp,
    )
    assert result.returncode != 0
    assert "WORK_ROOT basename" in result.stderr
    assert read_log(log) == []


def test_symlinked_work_root_refused(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    real = tmp / "realwork"
    real.mkdir()
    link = tmp / "worklink"
    link.symlink_to(real, target_is_directory=True)
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(link)],
        env,
        tmp,
    )
    assert result.returncode != 0
    assert "WORK_ROOT must not be a symlink" in result.stderr
    assert read_log(log) == []


def test_source_subtree_work_root_refused(shell_env):
    """A WORK_ROOT whose scene target resolves inside the repository is refused
    before any interpreter runs or any directory is created."""
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    # REPO_ROOT/tests is an existing real directory in the checkout; the scene
    # target beneath it is inside SCRIPT_DIR, so the destructive guard refuses.
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(REPO_ROOT / "tests")],
        env,
        tmp,
    )
    assert result.returncode != 0
    assert "inside the repository" in result.stderr
    assert read_log(log) == []
    assert not (REPO_ROOT / "tests" / "839920").exists()


# --- guarded root creation / provisioning ------------------------------------


def test_absent_work_root_created_plan_only(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    work = tmp / "sage3d_pointgoal"
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(work), "--plan-only"],
        env,
        tmp,
    )
    assert result.returncode == 0, result.stderr
    assert work.is_dir() and not work.is_symlink()
    assert (work / "839920" / "trajectories").is_dir()
    assert "DONE (plan only)" in result.stdout
    assert read_log(log) and "sage3d.cli.generate" in read_log(log)[0]


def test_output_root_provisioning_required(shell_env):
    """OUTPUT_ROOT must already exist; the pipeline never creates it."""
    tmp, env, log = shell_env
    missing = tmp / "must_exist"
    result = run_shell(
        ["839920", "--output-root", str(missing), "--plan-only"], env, tmp
    )
    assert result.returncode != 0
    assert "OUTPUT_ROOT must already exist" in result.stderr
    assert not missing.exists()
    assert read_log(log) == []


def test_plan_only_never_touches_output(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    work = tmp / "work"
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(work), "--plan-only"],
        env,
        tmp,
    )
    assert result.returncode == 0, result.stderr
    assert not (out / "839920").exists()
    assert read_log(log) and len(read_log(log)) == 1  # generate only


# --- allocator flow, force, validate -----------------------------------------


def test_full_flow_allocator_finalizer_validate(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    work = tmp / "work"
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(work)],
        env,
        tmp,
    )
    assert result.returncode == 0, result.stderr
    rendered = work / "839920" / "rendered"
    assert rendered.is_dir()
    assert (out / "839920").is_dir()
    assert "DONE:" in result.stdout

    calls = read_log(log)
    modules = [line.split()[1] for line in calls if line.startswith("-m ")]
    assert modules == [
        "sage3d.cli.generate",
        "sage3d.cli.create_staging",
        "sage3d.cli.render",
        "sage3d.cli.render",
        "sage3d.cli.finalize_render",
        "sage3d.cli.package",
    ]
    # package validate runs last as the check_package script.
    assert calls[-1].startswith(str(REPO_ROOT / "scripts" / "check_package.py"))
    assert "validate" in calls[-1]

    # Both render invocations used the same allocator-printed staging root,
    # and that root is a sibling of the final rendered dir.
    staging_line = next(
        line for line in calls if "sage3d.cli.create_staging" in line
    )
    staging_path = Path(staging_line.split("--final-target")[-1].strip())
    staging = staging_path.parent / (".rendered." + staging_path.name)
    assert staging.is_dir()
    rgb = next(line for line in calls if "--mode rgb" in line)
    depth = next(line for line in calls if "--mode depth" in line)
    assert f"--staging-root {staging.resolve()}" in rgb
    assert f"--staging-root {staging.resolve()}" in depth


def test_force_removes_existing_output(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    stale = out / "839920"
    stale.mkdir()
    (stale / "stale_marker").write_text("old")
    work = tmp / "work"
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(work), "--force"],
        env,
        tmp,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "839920").is_dir()
    assert not (out / "839920" / "stale_marker").exists()
    assert "Removing existing output" in result.stdout


def test_existing_output_without_force_refused(shell_env):
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    stale = out / "839920"
    stale.mkdir()
    (stale / "stale_marker").write_text("old")
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(tmp / "work")],
        env,
        tmp,
    )
    assert result.returncode != 0
    assert "Output already exists" in result.stdout
    assert (stale / "stale_marker").exists()  # untouched
    assert read_log(log) == []


def test_force_target_guard_symlink_refused(shell_env):
    """--force removal of a symlinked existing target is refused."""
    tmp, env, log = shell_env
    out = tmp / "out"
    out.mkdir()
    real_target = tmp / "real_target"
    real_target.mkdir()
    link = out / "839920"
    link.symlink_to(real_target, target_is_directory=True)
    work = tmp / "work"
    result = run_shell(
        ["839920", "--output-root", str(out), "--work-root", str(work), "--force"],
        env,
        tmp,
    )
    assert result.returncode != 0
    assert "SCENE_OUTPUT must not be a symlink" in result.stderr
    assert link.is_symlink()  # untouched


# --- legacy shim removal -----------------------------------------------------


def test_shell_uses_module_clis_not_legacy_scripts():
    """The pipeline shell must invoke module CLIs and the checker script, and
    must not call the legacy top-level scripts directly."""
    text = SHELL.read_text()
    for legacy in (
        "render_fisheye_sage3d.py",
        "package_lerobot_sage3d.py",
        "generate_sage3d_trajectories.py",
    ):
        assert legacy not in text
    for module in (
        "sage3d.cli.generate",
        "sage3d.cli.create_staging",
        "sage3d.cli.render",
        "sage3d.cli.finalize_render",
        "sage3d.cli.package",
    ):
        assert module in text
    assert "scripts/check_package.py" in text
    assert "--plan-only" in text
    assert "--force" in text
