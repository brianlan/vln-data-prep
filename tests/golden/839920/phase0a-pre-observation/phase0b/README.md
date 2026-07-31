# Phase 0b canonical baseline evidence

This directory is the review-sized evidence bundle for issue #6. The full
capture remains in the external artifact store because the rendered and
packaged outputs are too large for Git.

## Capture identity

- Baseline ID: `phase0a-pre-observation`
- Scene: `839920`
- Seed: `20260720`
- Captured candidate: `533f2a1508a9adad03b7d05d22d83b94247bd102`
- Full successful capture:
  `/ssd5/datasets/vln-fisheye/sage3d-canonical/phase0a-pre-observation/attempt-002-533f2a1`
- Capture host: NVIDIA GeForce RTX 5090, driver `580.65.06`
- Capture runtime: 873 seconds
- Full artifact size: 4,032,152,691 logical bytes; approximately 1.2 GiB
  physical storage because repeated inputs are hard-linked.

The committed bundle contains the capture and provenance reports, all checker
results, and the 15 selected RGB/depth frame pairs. Full generated artifacts
and 60 stage logs remain at the external path above.

## Machine-gate result

The machine gate passed:

- one designated baseline generate/render/package run passed;
- a second seeded generation produced the identical trajectory digest;
- five fresh-process characterization render pairs supplied the observations
  used to derive thresholds;
- two fresh-process held-out render/package runs passed without contributing
  to threshold derivation;
- all 25 mutation cases produced their expected outcome;
- all nine entries in `verification_manifest.json` are eligible;
- the baseline generation accepted five episodes after 18 attempts
  (12 rejected attempts, rejection rate 2/3).

Derived thresholds:

| Metric | Threshold |
| --- | ---: |
| `rgb_mask_leakage_mean_max` | 0.004230440172507115 |
| `rgb_masked_rmse` | 0.008801981015535369 |
| `rgb_masked_abs_error_p99` | 0.03352941176470591 |
| `depth_non_max_mask_iou` | 0.98 |
| `depth_error_p50` | 2 encoded units |
| `depth_error_p95` | 2 encoded units |
| `depth_error_p99` | 2 encoded units |

The capture retained warm filesystem and shader caches, while each RGB/depth
pair used fresh Isaac processes and a newly allocated output root. A cold-cache
or machine-restart characterization was not practical and is recorded as the
revision-8-permitted limitation.

## Diagnostics retained

The first full attempt is retained at:

`/ssd5/datasets/vln-fisheye/sage3d-canonical/phase0a-pre-observation/attempt-001-aee3d879`

It correctly failed when the `depth-missing-sentinel` mutation exposed that
only one live depth frame was being structurally validated. The checker now
validates every live depth frame. Two mutation preflights are also retained
under the external artifact root; the second records 25/25 expected outcomes.

## Validation

The capture and committed implementation were checked with:

```bash
PYTHONPATH="$PWD" "$SAGE3D_PACKAGE_PYTHON" -m pytest tests/package_safe
PYTHONPATH="$PWD" "$SAGE3D_ISAAC_PYTHON" -m pytest tests/isaac
PYTHONPATH="$PWD" SAGE3D_PHASE0B_CAPTURE=/ssd5/datasets/vln-fisheye/sage3d-canonical/phase0a-pre-observation/attempt-002-533f2a1 \
  "$SAGE3D_ISAAC_PYTHON" -m pytest -m sage3d_gpu tests/integration
```

## Approval status

The evidence is machine-approved but intentionally remains
`pending-independent-approval`. Per `../approval_ownership.md`, the Phase 0a
single-owner exception does not extend to the Phase 0b threshold report.

Merging this branch should be treated as the engineering owner's explicit
acceptance of:

1. the derived thresholds in `threshold_report.json`; and
2. the post-capture delivery re-estimate of 1.5 days for Phase 1, with
   low-to-medium residual risk.

If the merger is also the implementation author, that is an owner-approved
exception rather than an independent review and should be stated in the merge
record.
