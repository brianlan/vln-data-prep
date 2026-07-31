# Phase 0a validation lanes

This document records the three validation lanes defined by SAGE3D_REFACTOR_PLAN.md
revision 8 (decisions recorded before Phase 0a, item 1) and the interpreter
environment variables used to invoke them.

## Lanes

| Lane | Command | Runner | Assets |
| --- | --- | --- | --- |
| package-safe | `$SAGE3D_PACKAGE_PYTHON -m pytest tests/package_safe` | plain Python | none |
| Isaac-side | `$SAGE3D_ISAAC_PYTHON -m pytest tests/isaac` | Isaac Sim Python | as declared per test |
| pinned GPU / integration | `$SAGE3D_ISAAC_PYTHON -m pytest -m sage3d_gpu tests/integration` | Isaac Sim Python | external SAGE3D assets + GPU |

The package runner never imports Isaac-side modules (`cv2`, `scipy`,
`trimesh`, `pxr`, `isaacsim`). Test dependencies and pytest marker registration
are checked in during Phase 0a.

## Interpreter environment variables

Interpreters are referenced via env vars — no hardcoded `/ssd4/...` paths in
tests or source:

- `SAGE3D_ISAAC_PYTHON` — Isaac Sim Python (trajectory generation, rendering).
- `SAGE3D_PACKAGE_PYTHON` — plain Python (package-safe modules, artifact
  checkers, dataset packaging).

On the current development host these resolve to:

```bash
export SAGE3D_ISAAC_PYTHON=/ssd4/envs/isaac_sim_py311/bin/python
export SAGE3D_PACKAGE_PYTHON=/ssd4/envs/vln_data_prep_py311/bin/python
```

These are local defaults only; override the variables on other machines and do
not embed the `/ssd4` paths in source code or tests.

## Skip / gate policy

Developer and general CI runs may skip tests whose declared Isaac/GPU/assets
prerequisites are unavailable. The canonical `sage3d_gpu` validation lane
treats missing prerequisites as a **failure**, not a skip. If no hosted GPU CI
exists, its command output and provenance are attached as required PR
evidence.

Phase 0a green requires only the no-asset package-safe tests. Every later
Definition of Done that says "Phase 0b checks pass" requires evidence from the
canonical lane.

## Phase 0a green status

Phase 0a green is satisfied by:

```bash
$SAGE3D_PACKAGE_PYTHON -m pytest tests/package_safe
```

passing on the unmodified codebase, plus the tolerance policy reviewed and
immutable for the first capture attempt, all five approval owners recorded,
and an independent reviewer approving the policy (or the explicit exception
recorded in `approval_ownership.md`).