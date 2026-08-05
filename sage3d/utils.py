from dataclasses import dataclass


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
