#!/usr/bin/env python3
"""Generate deterministic, collision-aware PointGoal trajectories for SAGE3D.

.. deprecated::
    Use ``python -m sage3d.cli.generate`` instead.  This shim preserves the
    legacy command-line interface for existing callers.
"""

from __future__ import annotations

import sys
import warnings

from sage3d.cli.generate import main


if __name__ == "__main__":
    warnings.warn(
        "generate_sage3d_trajectories.py is deprecated; "
        "use 'python -m sage3d.cli.generate' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.exit(main())