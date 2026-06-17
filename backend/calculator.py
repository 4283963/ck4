import numpy as np
from typing import List, Dict, Tuple
from models import ShortestDistanceResult, OverlapResult


def euclidean_distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def interpolate_position(traj: np.ndarray, target_time: float) -> np.ndarray:
    timestamps = traj[:, 0]
    if target_time <= timestamps[0]:
        return traj[0, 1:4]
    if target_time >= timestamps[-1]:
        return traj[-1, 1:4]
    idx = np.searchsorted(timestamps, target_time) - 1
    idx = max(0, min(idx, len(timestamps) - 2))
    t0, t1 = timestamps[idx], timestamps[idx + 1]
    ratio = (target_time - t0) / (t1 - t0) if t1 != t0 else 0.0
    p0 = traj[idx, 1:4]
    p1 = traj[idx + 1, 1:4]
    return p0 + ratio * (p1 - p0)


def compute_shortest_distance_between_pair(
    traj_a: np.ndarray, traj_a_id: str,
    traj_b: np.ndarray, traj_b_id: str,
    sample_rate: float = 0.1
) -> ShortestDistanceResult:
    t_min = max(traj_a[0, 0], traj_b[0, 0])
    t_max = min(traj_a[-1, 0], traj_b[-1, 0])
    if t_min >= t_max:
        return ShortestDistanceResult(
            drone_a=traj_a_id, drone_b=traj_b_id,
            distance=float('inf'), timestamp=t_min,
            point_a=[0, 0, 0], point_b=[0, 0, 0]
        )

    timestamps = np.arange(t_min, t_max + sample_rate, sample_rate)
    min_dist = float('inf')
    min_t = t_min
    min_pa = np.zeros(3)
    min_pb = np.zeros(3)

    for t in timestamps:
        pa = interpolate_position(traj_a, t)
        pb = interpolate_position(traj_b, t)
        dist = euclidean_distance_3d(pa, pb)
        if dist < min_dist:
            min_dist = dist
            min_t = t
            min_pa = pa
            min_pb = pb

    return ShortestDistanceResult(
        drone_a=traj_a_id, drone_b=traj_b_id,
        distance=min_dist, timestamp=float(min_t),
        point_a=min_pa.tolist(), point_b=min_pb.tolist()
    )


def compute_all_shortest_distances(
    trajectories: Dict[str, np.ndarray]
) -> List[ShortestDistanceResult]:
    ids = list(trajectories.keys())
    results = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            result = compute_shortest_distance_between_pair(
                trajectories[ids[i]], ids[i],
                trajectories[ids[j]], ids[j]
            )
            results.append(result)
    results.sort(key=lambda r: r.distance)
    return results


def compute_trajectory_overlap(
    traj_a: np.ndarray, traj_a_id: str,
    traj_b: np.ndarray, traj_b_id: str,
    threshold: float = 50.0
) -> OverlapResult:
    all_times_a = traj_a[:, 0]
    all_times_b = traj_b[:, 0]
    t_min = max(all_times_a[0], all_times_b[0])
    t_max = min(all_times_a[-1], all_times_b[-1])
    if t_min >= t_max:
        return OverlapResult(
            drone_a=traj_a_id, drone_b=traj_b_id,
            overlap_ratio=0.0, overlap_points_count=0,
            total_points_count=0
        )

    sample_times = np.linspace(t_min, t_max, num=100)
    overlap_count = 0
    for t in sample_times:
        pa = interpolate_position(traj_a, t)
        pb = interpolate_position(traj_b, t)
        dist = euclidean_distance_3d(pa, pb)
        if dist <= threshold:
            overlap_count += 1

    total_count = len(sample_times)
    ratio = overlap_count / total_count if total_count > 0 else 0.0
    return OverlapResult(
        drone_a=traj_a_id, drone_b=traj_b_id,
        overlap_ratio=float(ratio),
        overlap_points_count=overlap_count,
        total_points_count=total_count
    )


def compute_all_overlaps(
    trajectories: Dict[str, np.ndarray],
    threshold: float = 50.0
) -> List[OverlapResult]:
    ids = list(trajectories.keys())
    results = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            result = compute_trajectory_overlap(
                trajectories[ids[i]], ids[i],
                trajectories[ids[j]], ids[j],
                threshold
            )
            results.append(result)
    results.sort(key=lambda r: r.overlap_ratio, reverse=True)
    return results


def get_overlap_region_points(
    traj_a: np.ndarray, traj_b: np.ndarray,
    threshold: float = 50.0,
    sample_rate: float = 0.5
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    t_min = max(traj_a[0, 0], traj_b[0, 0])
    t_max = min(traj_a[-1, 0], traj_b[-1, 0])
    if t_min >= t_max:
        return []
    timestamps = np.arange(t_min, t_max + sample_rate, sample_rate)
    overlap_points = []
    for t in timestamps:
        pa = interpolate_position(traj_a, t)
        pb = interpolate_position(traj_b, t)
        dist = euclidean_distance_3d(pa, pb)
        if dist <= threshold:
            overlap_points.append((pa, pb, float(t)))
    return overlap_points


def get_time_range(trajectories: Dict[str, np.ndarray]) -> Tuple[float, float]:
    if not trajectories:
        return 0.0, 1.0
    all_t_min = min(t[0, 0] for t in trajectories.values())
    all_t_max = max(t[-1, 0] for t in trajectories.values())
    return float(all_t_min), float(all_t_max)
