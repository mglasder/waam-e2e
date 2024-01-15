from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from nest.engineering.ddd import Aggregate, ValueObject
from nest.engineering.geometry import Point2, Point3


class TorchPosition(ValueObject):
    """Coordinates are in global coordinate system (parallel to yz-plane)."""

    global_y_mm: float
    global_z_mm: float

    global_y_idx: int
    global_z_idx: int


class FootprintEdge(ValueObject):
    left: Optional[Point2]
    left_idx: Optional[int]
    right: Optional[Point2]
    right_idx: Optional[int]


class Mesh3D(ValueObject):
    stl: Path


class Mesh2D(ValueObject):
    points: List[Point2]

    @property
    def xs(self) -> np.ndarray:
        return np.asarray([p.x[0] for p in self.points])

    @property
    def ys(self) -> np.ndarray:
        return np.asarray([p.y[0] for p in self.points])

    @property
    def xys(self) -> np.ndarray:
        return np.asarray([np.array([p.x[0], p.y[0]]) for p in self.points])


class CrossSectionSample(Aggregate):
    processing_history: List[Tuple[str, str]] = []

    experiment: str
    bead_id: int
    welding_params: Dict[str, Any]

    mesh_before: Mesh3D
    mesh_after: Mesh3D

    slice_start: Optional[Point3]
    slice_stop: Optional[Point3]
    resolution: Optional[float]

    slices_before: Optional[List[Mesh2D]]
    slices_after: Optional[List[Mesh2D]]

    slice_averaged_before: Optional[Mesh2D]
    slice_averaged_after: Optional[Mesh2D]

    slice_std_after: Optional[Mesh2D]
    slice_std_before: Optional[Mesh2D]

    slice_aligned_before: Optional[Mesh2D]
    slice_aligned_after: Optional[Mesh2D]

    slice_based_before: Optional[Mesh2D]
    slice_based_after: Optional[Mesh2D]

    before_image: Optional[np.ndarray]
    after_image: Optional[np.ndarray]

    footprint: Optional[FootprintEdge]
    footprint_based: Optional[FootprintEdge]
    footprint_after: Optional[FootprintEdge]

    area: Optional[float]
    torchposition: Optional[TorchPosition]
