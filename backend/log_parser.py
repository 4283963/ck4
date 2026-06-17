import csv
import json
import os
from typing import List, Dict
from models import DronePoint, DroneTrajectory
import numpy as np


def parse_csv_log(file_path: str) -> DroneTrajectory:
    drone_id = os.path.splitext(os.path.basename(file_path))[0]
    points = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = DronePoint(
                timestamp=float(row['timestamp']),
                x=float(row['x']),
                y=float(row['y']),
                z=float(row['z']),
                battery=float(row['battery'])
            )
            points.append(point)
    points.sort(key=lambda p: p.timestamp)
    return DroneTrajectory(drone_id=drone_id, points=points)


def parse_json_log(file_path: str) -> DroneTrajectory:
    drone_id = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'r') as f:
        data = json.load(f)
    points = [DronePoint(**p) for p in data.get('points', data)]
    points.sort(key=lambda p: p.timestamp)
    return DroneTrajectory(drone_id=drone_id, points=points)


def load_logs_from_dir(log_dir: str) -> List[DroneTrajectory]:
    trajectories = []
    if not os.path.exists(log_dir):
        return trajectories
    for filename in sorted(os.listdir(log_dir)):
        filepath = os.path.join(log_dir, filename)
        if filename.endswith('.csv'):
            trajectories.append(parse_csv_log(filepath))
        elif filename.endswith('.json'):
            trajectories.append(parse_json_log(filepath))
    return trajectories


def trajectory_to_numpy(trajectory: DroneTrajectory) -> np.ndarray:
    return np.array([[p.timestamp, p.x, p.y, p.z, p.battery] for p in trajectory.points])


def all_trajectories_to_numpy(trajectories: List[DroneTrajectory]) -> Dict[str, np.ndarray]:
    return {t.drone_id: trajectory_to_numpy(t) for t in trajectories}
