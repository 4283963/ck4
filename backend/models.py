from pydantic import BaseModel
from typing import List, Optional
import numpy as np


class DronePoint(BaseModel):
    timestamp: float
    x: float
    y: float
    z: float
    battery: float


class DroneTrajectory(BaseModel):
    drone_id: str
    points: List[DronePoint]


class ShortestDistanceResult(BaseModel):
    drone_a: str
    drone_b: str
    distance: float
    timestamp: float
    point_a: List[float]
    point_b: List[float]


class OverlapResult(BaseModel):
    drone_a: str
    drone_b: str
    overlap_ratio: float
    overlap_points_count: int
    total_points_count: int


class AnalysisResponse(BaseModel):
    trajectories: List[DroneTrajectory]
    shortest_distances: List[ShortestDistanceResult]
    overlaps: List[OverlapResult]
    time_range: List[float]


class FrameData(BaseModel):
    timestamp: float
    drone_positions: dict
    current_distances: List[ShortestDistanceResult]
