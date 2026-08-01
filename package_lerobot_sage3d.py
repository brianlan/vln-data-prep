#!/usr/bin/env python3
"""Rollout shim for the legacy SAGE3D package script.

Deprecated. This script now delegates to the production module CLI
(``python -m sage3d.cli.package``) so legacy callers keep working with the
same argument surface and exit behavior while the package pipeline becomes
non-destructive (sibling staging, staged validation, atomic rename onto an
absent target).

Legacy callers that relied on the script overwriting an existing output
directory must clear the target themselves (or pass through the shell's
``--force`` path) because the production pipeline refuses an existing final
target.
"""

from __future__ import annotations

import sys

from sage3d.cli.package import main

if __name__ == "__main__":
    main(sys.argv[1:])
