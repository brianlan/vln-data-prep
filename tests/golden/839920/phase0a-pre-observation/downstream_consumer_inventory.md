# Phase 0a downstream-consumer inventory

This inventory records every actual in-repo or external downstream consumer of
the SAGE3D packaged dataset that is available at Phase 0a, per
SAGE3D_REFACTOR_PLAN.md revision 8 (Packaging compatibility policy). When no
such consumer exists, that is recorded explicitly rather than left as an
implementer choice.

## In-repo consumers

| Consumer | What it reads | SAGE3D-authoritative? | Notes |
| --- | --- | --- | --- |
| `prepare_trajectories.py` | parquet `observation.camera_extrinsic`, `action` | no — supplemental smoke only | Reads only shared-Parquet schema fields, not SAGE3D point-goal, render, or metadata contracts. Phase 5 runs it as a supplemental smoke; failure is actionable but it does not replace the SAGE3D package oracle. |
| `package_lerobot.py` | generic (non-SAGE3D) parquet + meta + images | no — out of scope | Generic LeRobot packager for non-SAGE3D datasets. Not folded into `sage3d/` and remains behaviorally untouched. Not a SAGE3D consumer. |

## External consumers

None found at Phase 0a. No external downstream consumer of the SAGE3D
packaged dataset is currently available.

## Binding contract in the absence of a named consumer

Per the plan, the package oracle is the binding compatibility contract when no
named consumer exists. The lack of a named consumer is recorded here, not left
as an implementer choice. Phase 5 adds a checked-in format note and a smoke
test for the named consumer when one is found.

## Update policy

If a real downstream consumer is found later, its pinned smoke test becomes
required at Phase 5. Converting to standard LeRobot is a separate post-refactor
migration with its own output-version boundary and data regeneration plan.