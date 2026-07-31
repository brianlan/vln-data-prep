## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Development Environment

SAGE3D work uses separate environments rather than one all-purpose Python
environment:

- `SAGE3D_ISAAC_PYTHON`: Isaac Sim Python for trajectory generation and
  rendering. Rendering also requires a compatible NVIDIA GPU and the external
  SAGE3D assets.
- `SAGE3D_PACKAGE_PYTHON`: plain Python for package-safe modules, artifact
  checkers, and dataset packaging. This lane must not import Isaac, `pxr`,
  OpenCV, SciPy, or trimesh unless a task explicitly belongs to the Isaac lane.

On the current development host, use these verified Python 3.11.15
interpreters:

```bash
export SAGE3D_ISAAC_PYTHON=/ssd4/envs/isaac_sim_py311/bin/python
export SAGE3D_PACKAGE_PYTHON=/ssd4/envs/vln_data_prep_py311/bin/python
```

They resolve to the corresponding `bin/python3.11` executables. Treat these as
local defaults: override the variables on other machines and do not embed the
`/ssd4` paths in source code or tests. The package is intentionally run from
the repository checkout, so invoke modules with the repository root prepended
to `PYTHONPATH`.

The planned validation lanes are:

```bash
$SAGE3D_PACKAGE_PYTHON -m pytest tests/package_safe
$SAGE3D_ISAAC_PYTHON -m pytest tests/isaac
$SAGE3D_ISAAC_PYTHON -m pytest -m sage3d_gpu tests/integration
```

Package-safe tests should run without GPU access or external SAGE3D assets.
Isaac/GPU tests require their declared runtime and assets; consult
`SAGE3D_REFACTOR_PLAN.md` for the current phase-specific prerequisites.
