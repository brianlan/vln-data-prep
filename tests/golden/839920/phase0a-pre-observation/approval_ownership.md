# Phase 0a approval ownership record

This record names the accountable owners required by the Phase 0 decision
record (SAGE3D_REFACTOR_PLAN.md revision 8, decision 9). At least one approver
other than the implementation author is required for the items marked
*independent review required*. When staffing makes independent review
impossible, an explicit owner-approved exception with scope and rationale is
recorded below; one-person approval is not the default policy.

## Approval roles

| Role | Owner | Independent review required |
| --- | --- | --- |
| Baseline / tolerance approval | brianlan | yes |
| Canonical GPU evidence | brianlan | yes |
| Re-baselining | brianlan | yes |
| Package-format sign-off | brianlan | yes |
| Compatibility-shim removal sign-off | brianlan | yes |

## Independent-review exception

At Phase 0a the repository has a single available collaborator (`brianlan`),
who is both the implementation author and the owner of every approval role
above. Full independent review is therefore impossible at this time. Per the
plan's explicit exception clause, this is recorded here rather than left
implicit:

- **Exception scope:** Phase 0a pre-observation tolerance policy only.
- **Rationale:** no second collaborator with the required domain knowledge is
  currently available. The tolerance policy is pre-observation and contains no
  empirically derived thresholds, so the risk of a self-approved mistake
  affecting downstream captures is bounded by the explicit protocol above.
- **Lift condition:** when a second qualified approver is available, every
  later gate (Phase 0b threshold report, re-baselining, Phase 4 render
  extraction, package-format compatibility sign-off) must receive independent
  approval. This Phase 0a exception does not extend to those later gates.

This exception is owner-approved and recorded as required by the plan.