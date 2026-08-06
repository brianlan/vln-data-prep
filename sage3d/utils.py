from functools import partial
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MapTransform:
    height: int
    width: int
    scale: float
    lower_x: float
    lower_y: float

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        # Raw occupancy columns run opposite world +X.
        x = self.lower_x + (self.width - col - 0.5) * self.scale
        # Raw rows already run with world +Y.
        y = self.lower_y + (row + 0.5) * self.scale
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        canonical_col = int(round((x - self.lower_x) / self.scale - 0.5))
        col = self.width - 1 - canonical_col
        row = int(round((y - self.lower_y) / self.scale - 0.5))
        return row, col


def ensured_path(input, ensure_parent=False):
    """Often used in the scenario that the path we want to write things to is ensured to be exist."""
    p = Path(input)
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


parent_ensured_path = partial(ensured_path, ensure_parent=True)
