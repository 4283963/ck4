import numpy as np
from typing import List, Dict, Tuple
from models import ShortestDistanceResult, OverlapResult

EPS = 1e-8
SAFE_EPS = 1e-6


def _safe_value(val: np.ndarray, default: float = 0.0) -> np.ndarray:
    return np.where(np.isfinite(val), val, default)


def euclidean_distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    diff = _safe_value(p1) - _safe_value(p2)
    sq_sum = np.sum(diff ** 2)
    if not np.isfinite(sq_sum) or sq_sum < 0:
        return 0.0
    return float(np.sqrt(sq_sum))


def interpolate_position(traj: np.ndarray, target_time: float) -> np.ndarray:
    if traj is None or len(traj) == 0:
        return np.zeros(3)

    traj = _safe_value(traj)
    timestamps = traj[:, 0]

    if len(timestamps) == 1:
        return _safe_value(traj[0, 1:4])

    if target_time <= timestamps[0] + EPS:
        return _safe_value(traj[0, 1:4])
    if target_time >= timestamps[-1] - EPS:
        return _safe_value(traj[-1, 1:4])

    idx = np.searchsorted(timestamps, target_time) - 1
    idx = max(0, min(idx, len(timestamps) - 2))
    t0, t1 = float(timestamps[idx]), float(timestamps[idx + 1])
    dt = t1 - t0

    if abs(dt) < SAFE_EPS:
        return _safe_value(traj[idx, 1:4])

    ratio = (target_time - t0) / dt
    ratio = max(0.0, min(1.0, ratio))

    p0 = _safe_value(traj[idx, 1:4])
    p1 = _safe_value(traj[idx + 1, 1:4])
    result = p0 + ratio * (p1 - p0)
    return _safe_value(result)


def compute_shortest_distance_between_pair(
    traj_a: np.ndarray, traj_a_id: str,
    traj_b: np.ndarray, traj_b_id: str,
    sample_rate: float = 0.1
) -> ShortestDistanceResult:
    if traj_a is None or traj_b is None or len(traj_a) == 0 or len(traj_b) == 0:
        return ShortestDistanceResult(
            drone_a=traj_a_id, drone_b=traj_b_id,
            distance=float('inf'), timestamp=0.0,
            point_a=[0, 0, 0], point_b=[0, 0, 0]
        )

    traj_a = _safe_value(traj_a)
    traj_b = _safe_value(traj_b)

    t_min = max(float(traj_a[0, 0]), float(traj_b[0, 0]))
    t_max = min(float(traj_a[-1, 0]), float(traj_b[-1, 0]))

    if t_min + SAFE_EPS >= t_max:
        return ShortestDistanceResult(
            drone_a=traj_a_id, drone_b=traj_b_id,
            distance=float('inf'), timestamp=t_min,
            point_a=[0, 0, 0], point_b=[0, 0, 0]
        )

    if sample_rate < SAFE_EPS:
        sample_rate = 0.1

    n_samples = int((t_max - t_min) / sample_rate) + 1
    n_samples = max(2, min(n_samples, 10000))
    timestamps = np.linspace(t_min, t_max, num=n_samples)

    min_dist = float('inf')
    min_t = t_min
    min_pa = np.zeros(3)
    min_pb = np.zeros(3)

    for t in timestamps:
        pa = interpolate_position(traj_a, float(t))
        pb = interpolate_position(traj_b, float(t))
        dist = euclidean_distance_3d(pa, pb)
        if np.isfinite(dist) and dist < min_dist:
            min_dist = dist
            min_t = float(t)
            min_pa = pa
            min_pb = pb

    if not np.isfinite(min_dist):
        min_dist = 0.0

    return ShortestDistanceResult(
        drone_a=traj_a_id, drone_b=traj_b_id,
        distance=min_dist, timestamp=float(min_t),
        point_a=_safe_value(min_pa).tolist(),
        point_b=_safe_value(min_pb).tolist()
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
    if traj_a is None or traj_b is None or len(traj_a) == 0 or len(traj_b) == 0:
        return OverlapResult(
            drone_a=traj_a_id, drone_b=traj_b_id,
            overlap_ratio=0.0, overlap_points_count=0,
            total_points_count=0
        )

    traj_a = _safe_value(traj_a)
    traj_b = _safe_value(traj_b)

    all_times_a = traj_a[:, 0]
    all_times_b = traj_b[:, 0]
    t_min = max(float(all_times_a[0]), float(all_times_b[0]))
    t_max = min(float(all_times_a[-1]), float(all_times_b[-1]))

    if t_min + SAFE_EPS >= t_max:
        return OverlapResult(
            drone_a=traj_a_id, drone_b=traj_b_id,
            overlap_ratio=0.0, overlap_points_count=0,
            total_points_count=0
        )

    n_samples = 100
    sample_times = np.linspace(t_min, t_max, num=n_samples)
    overlap_count = 0
    total_count = 0

    for t in sample_times:
        pa = interpolate_position(traj_a, float(t))
        pb = interpolate_position(traj_b, float(t))
        dist = euclidean_distance_3d(pa, pb)
        if np.isfinite(dist) and dist <= threshold:
            overlap_count += 1
        total_count += 1

    ratio = overlap_count / total_count if total_count > 0 else 0.0
    if not np.isfinite(ratio):
        ratio = 0.0

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
    if traj_a is None or traj_b is None or len(traj_a) == 0 or len(traj_b) == 0:
        return []

    traj_a = _safe_value(traj_a)
    traj_b = _safe_value(traj_b)

    t_min = max(float(traj_a[0, 0]), float(traj_b[0, 0]))
    t_max = min(float(traj_a[-1, 0]), float(traj_b[-1, 0]))

    if t_min + SAFE_EPS >= t_max:
        return []

    if sample_rate < SAFE_EPS:
        sample_rate = 0.5

    n_samples = int((t_max - t_min) / sample_rate) + 1
    n_samples = max(2, min(n_samples, 5000))
    timestamps = np.linspace(t_min, t_max, num=n_samples)

    overlap_points = []
    for t in timestamps:
        pa = interpolate_position(traj_a, float(t))
        pb = interpolate_position(traj_b, float(t))
        dist = euclidean_distance_3d(pa, pb)
        if np.isfinite(dist) and dist <= threshold:
            overlap_points.append((_safe_value(pa), _safe_value(pb), float(t)))
    return overlap_points


def get_time_range(trajectories: Dict[str, np.ndarray]) -> Tuple[float, float]:
    if not trajectories:
        return 0.0, 1.0
    all_t_min = min(t[0, 0] for t in trajectories.values())
    all_t_max = max(t[-1, 0] for t in trajectories.values())
    return float(all_t_min), float(all_t_max)


BEAUFORT_SPEED_MAP = {
    0: 0.0, 1: 0.3, 2: 1.6, 3: 3.4, 4: 5.5,
    5: 8.0, 6: 10.8, 7: 13.9, 8: 17.2, 9: 20.8,
    10: 24.5, 11: 28.5, 12: 32.7
}


def beaufort_to_mps(level: float) -> float:
    level = max(0.0, min(12.0, float(level)))
    lo = int(level)
    hi = min(lo + 1, 12)
    frac = level - lo
    speed_lo = BEAUFORT_SPEED_MAP.get(lo, 0.0)
    speed_hi = BEAUFORT_SPEED_MAP.get(hi, speed_lo)
    return speed_lo + frac * (speed_hi - speed_lo)


def compute_wind_vector(wind_angle_deg: float, wind_level: float) -> np.ndarray:
    wind_speed_mps = beaufort_to_mps(wind_level)
    if wind_speed_mps < SAFE_EPS:
        return np.zeros(3)
    angle_rad = np.deg2rad(float(wind_angle_deg))
    dx = wind_speed_mps * np.cos(angle_rad)
    dy = wind_speed_mps * np.sin(angle_rad)
    dz = wind_speed_mps * 0.05
    return np.array([dx, dy, dz])


def apply_wind_to_trajectory(
    traj: np.ndarray,
    wind_vector: np.ndarray,
    t_start: float
) -> np.ndarray:
    if traj is None or len(traj) == 0:
        return np.zeros((0, 5))

    traj = _safe_value(traj.copy())
    wind_vector = _safe_value(wind_vector)

    if np.all(np.abs(wind_vector) < SAFE_EPS):
        return traj

    for i in range(len(traj)):
        t = float(traj[i, 0])
        dt = max(0.0, t - t_start)
        drift = wind_vector * dt
        traj[i, 1] += drift[0]
        traj[i, 2] += drift[1]
        traj[i, 3] += drift[2]

    return _safe_value(traj)


def apply_wind_to_all_trajectories(
    trajectories: Dict[str, np.ndarray],
    wind_angle_deg: float,
    wind_level: float
) -> Dict[str, np.ndarray]:
    wind_vector = compute_wind_vector(wind_angle_deg, wind_level)
    if np.all(np.abs(wind_vector) < SAFE_EPS):
        return trajectories

    all_times = []
    for traj in trajectories.values():
        if len(traj) > 0:
            all_times.append(float(traj[0, 0]))
    t_start = min(all_times) if all_times else 0.0

    result = {}
    for did, traj in trajectories.items():
        result[did] = apply_wind_to_trajectory(traj, wind_vector, t_start)
    return result
