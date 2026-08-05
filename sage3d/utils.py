from dataclasses import dataclass


@dataclass(frozen=True)
class MapTransform:
    height: int
    width: int
    scale: float
    lower_x: float
    lower_y: float

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.lower_x + (col + 0.5) * self.scale
        # Raw InteriorGS occupancy maps use row 0 at the lower world-Y bound.
        # SAGE3D's semantic-map export flips the raw occupancy image for
        # visualization, but that flip must not be applied while planning
        # directly on occupancy.png.
        y = self.lower_y + (row + 0.5) * self.scale
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = int(round((x - self.lower_x) / self.scale - 0.5))
        row = int(round((y - self.lower_y) / self.scale - 0.5))
        return row, col