import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calculator import (
    interpolate_position,
    euclidean_distance_3d,
    compute_shortest_distance_between_pair,
    compute_trajectory_overlap,
    get_overlap_region_points,
    _safe_value
)

print("=" * 70)
print("除零 Bug 修复验证测试")
print("=" * 70)

passed = 0
failed = 0

def test_case(name, func):
    global passed, failed
    try:
        func()
        print(f"✓ PASS: {name}")
        passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {name} - {e}")
        failed += 1
    except Exception as e:
        print(f"✗ ERROR: {name} - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

def test_zero_distance():
    """测试：两架无人机坐标完全重合"""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0])
    dist = euclidean_distance_3d(p1, p2)
    assert isinstance(dist, float), f"距离类型错误: {type(dist)}"
    assert abs(dist - 0.0) < 1e-10, f"距离不为0: {dist}"
    assert not np.isnan(dist), f"距离为NaN"
    assert np.isfinite(dist), f"距离非有限值"

def test_identical_timestamps():
    """测试：日志中有相同的时间戳（除零风险最高的场景）"""
    traj = np.array([
        [0.0, 10.0, 20.0, 30.0, 90.0],
        [0.0, 15.0, 25.0, 35.0, 89.0],
        [1.0, 20.0, 30.0, 40.0, 88.0]
    ])
    result = interpolate_position(traj, 0.0)
    assert result.shape == (3,), f"结果形状错误: {result.shape}"
    assert not np.any(np.isnan(result)), f"结果包含NaN: {result}"
    assert np.all(np.isfinite(result)), f"结果非有限: {result}"

def test_very_small_timestamp_diff():
    """测试：时间差极小（1e-15）的插值"""
    traj = np.array([
        [0.0, 10.0, 20.0, 30.0, 90.0],
        [1e-15, 15.0, 25.0, 35.0, 89.0],
        [1.0, 20.0, 30.0, 40.0, 88.0]
    ])
    result = interpolate_position(traj, 5e-16)
    assert not np.any(np.isnan(result)), f"极小时间差结果包含NaN: {result}"
    assert np.all(np.isfinite(result)), f"极小时间差结果非有限: {result}"

def test_shortest_distance_with_overlap():
    """测试：两架无人机在某一时刻完全重合"""
    traj_a = np.array([
        [0.0, 0.0, 0.0, 0.0, 100.0],
        [1.0, 10.0, 10.0, 10.0, 99.0],
        [2.0, 20.0, 20.0, 20.0, 98.0]
    ])
    traj_b = np.array([
        [0.0, 5.0, 5.0, 5.0, 100.0],
        [1.0, 10.0, 10.0, 10.0, 99.0],
        [2.0, 15.0, 15.0, 15.0, 98.0]
    ])
    result = compute_shortest_distance_between_pair(traj_a, "A", traj_b, "B")
    assert not np.isnan(result.distance), f"距离为NaN"
    assert np.isfinite(result.distance), f"距离非有限: {result.distance}"
    assert abs(result.distance) < 1e-10, f"重合点距离不为0: {result.distance}"
    assert all(not np.isnan(x) for x in result.point_a), f"point_a 含NaN"
    assert all(not np.isnan(x) for x in result.point_b), f"point_b 含NaN"

def test_overlap_with_identical_trajectories():
    """测试：两架无人机轨迹完全相同"""
    traj_a = np.array([
        [0.0, 0.0, 0.0, 0.0, 100.0],
        [1.0, 10.0, 10.0, 10.0, 99.0],
        [2.0, 20.0, 20.0, 20.0, 98.0]
    ])
    result = compute_trajectory_overlap(traj_a, "A", traj_a, "A", threshold=50.0)
    assert not np.isnan(result.overlap_ratio), f"重叠度为NaN"
    assert np.isfinite(result.overlap_ratio), f"重叠度非有限: {result.overlap_ratio}"
    assert abs(result.overlap_ratio - 1.0) < 1e-10, f"完全相同轨迹重叠度不为1: {result.overlap_ratio}"

def test_nan_input_protection():
    """测试：输入包含 NaN 时的保护"""
    traj = np.array([
        [0.0, np.nan, 20.0, 30.0, 90.0],
        [1.0, 10.0, np.inf, 35.0, 89.0],
        [2.0, 20.0, 30.0, -np.inf, 88.0]
    ])
    safe_traj = _safe_value(traj)
    assert not np.any(np.isnan(safe_traj)), f"_safe_value 未清除NaN"
    assert not np.any(np.isinf(safe_traj)), f"_safe_value 未清除Inf"

    result = interpolate_position(traj, 0.5)
    assert not np.any(np.isnan(result)), f"NaN输入导致NaN输出: {result}"
    assert np.all(np.isfinite(result)), f"NaN输入导致非有限输出: {result}"

def test_empty_trajectory():
    """测试：空轨迹输入"""
    traj_a = np.zeros((0, 5))
    traj_b = np.array([[0.0, 0.0, 0.0, 0.0, 100.0]])
    result1 = compute_shortest_distance_between_pair(traj_a, "A", traj_b, "B")
    assert not np.isnan(result1.distance), "空轨迹导致NaN"

    result2 = compute_trajectory_overlap(traj_a, "A", traj_b, "B")
    assert not np.isnan(result2.overlap_ratio), "空轨迹导致重叠度NaN"

def test_zero_time_range():
    """测试：两架无人机没有时间重叠"""
    traj_a = np.array([
        [0.0, 0.0, 0.0, 0.0, 100.0],
        [1.0, 10.0, 10.0, 10.0, 99.0]
    ])
    traj_b = np.array([
        [2.0, 20.0, 20.0, 20.0, 100.0],
        [3.0, 30.0, 30.0, 30.0, 99.0]
    ])
    result = compute_shortest_distance_between_pair(traj_a, "A", traj_b, "B")
    assert not np.isnan(result.distance), "无时间重叠导致NaN"
    assert result.distance == float('inf') or np.isfinite(result.distance), f"距离异常: {result.distance}"

def test_overlap_region_with_zero_distance():
    """测试：获取重叠区域，包含零距离点"""
    traj_a = np.array([
        [0.0, 0.0, 0.0, 0.0, 100.0],
        [1.0, 10.0, 10.0, 10.0, 99.0],
        [2.0, 20.0, 20.0, 20.0, 98.0]
    ])
    traj_b = np.array([
        [0.0, 0.0, 0.0, 0.0, 100.0],
        [1.0, 10.0, 10.0, 10.0, 99.0],
        [2.0, 20.0, 20.0, 20.0, 98.0]
    ])
    regions = get_overlap_region_points(traj_a, traj_b, threshold=100.0)
    assert len(regions) > 0, "完全相同轨迹应有重叠区域"
    for pa, pb, t in regions:
        assert not np.any(np.isnan(pa)), f"重叠区域点pa含NaN"
        assert not np.any(np.isnan(pb)), f"重叠区域点pb含NaN"
        dist = euclidean_distance_3d(pa, pb)
        assert np.isfinite(dist), f"重叠区域距离非有限"

def test_single_point_trajectory():
    """测试：只有一个采样点的轨迹"""
    traj = np.array([[5.0, 100.0, 200.0, 300.0, 80.0]])
    result = interpolate_position(traj, 5.0)
    assert not np.any(np.isnan(result)), f"单点轨迹插值含NaN"
    assert np.allclose(result, [100.0, 200.0, 300.0]), f"单点轨迹插值错误: {result}"

    result2 = interpolate_position(traj, 0.0)
    assert np.allclose(result2, [100.0, 200.0, 300.0]), "边界插值错误"

    result3 = interpolate_position(traj, 10.0)
    assert np.allclose(result3, [100.0, 200.0, 300.0]), "边界插值错误"

print("\n--- 开始测试 ---\n")

test_case("坐标完全重合的距离计算", test_zero_distance)
test_case("相同时间戳的插值", test_identical_timestamps)
test_case("极小时间差(1e-15)的插值", test_very_small_timestamp_diff)
test_case("重合点的最短协同距离计算", test_shortest_distance_with_overlap)
test_case("完全相同轨迹的重叠度计算", test_overlap_with_identical_trajectories)
test_case("NaN/Inf 输入的防护", test_nan_input_protection)
test_case("空轨迹输入保护", test_empty_trajectory)
test_case("无时间重叠的处理", test_zero_time_range)
test_case("含零距离的重叠区域计算", test_overlap_region_with_zero_distance)
test_case("单点轨迹插值", test_single_point_trajectory)

print(f"\n{'=' * 70}")
print(f"测试完成: 通过 {passed} 个，失败 {failed} 个")
print("=" * 70)

if failed == 0:
    print("\n🎉 所有测试通过！除零 Bug 已修复。")
    sys.exit(0)
else:
    print(f"\n⚠️  有 {failed} 个测试失败，需要进一步修复。")
    sys.exit(1)
