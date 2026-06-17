import os
import csv
import json
import sys
import requests

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG_DIR = os.path.abspath(LOG_DIR)

API_BASE = "http://localhost:8088"

def generate_collision_logs():
    """生成包含完全重合坐标的测试日志"""

    with open(os.path.join(LOG_DIR, "test_drone_A.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z", "battery"])
        for t in range(10):
            if t == 5:
                writer.writerow([float(t), 100.0, 100.0, 50.0, 90.0 - t])
            else:
                writer.writerow([float(t), 50.0 + t * 10, 50.0 + t * 5, 30.0 + t * 2, 90.0 - t])

    with open(os.path.join(LOG_DIR, "test_drone_B.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z", "battery"])
        for t in range(10):
            if t == 5:
                writer.writerow([float(t), 100.0, 100.0, 50.0, 85.0 - t])
            else:
                writer.writerow([float(t), 150.0 - t * 10, 100.0 - t * 5, 60.0 - t * 2, 85.0 - t])

    with open(os.path.join(LOG_DIR, "test_drone_C.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z", "battery"])
        for t in range(10):
            if t == 5:
                writer.writerow([float(t), 100.0, 100.0, 50.0, 80.0 - t])
            else:
                writer.writerow([float(t), 80.0 + t * 3, 120.0 + t * 4, 40.0 + t * 1.5, 80.0 - t])

    print("生成测试日志: test_drone_A.csv, test_drone_B.csv, test_drone_C.csv")
    print("  -> 三架无人机在 t=5s 时坐标完全重合 (100, 100, 50)")

def test_api():
    """测试完整 API 调用链"""

    print(f"\n{'=' * 70}")
    print("端到端 API 测试 (包含重合坐标场景)")
    print("=" * 70)

    try:
        print("\n1. 测试 /api/analyze 接口...")
        r = requests.get(f"{API_BASE}/api/analyze?overlap_threshold=50", timeout=10)
        assert r.status_code == 200, f"HTTP 错误: {r.status_code}"
        data = r.json()

        trajectories = data['trajectories']
        print(f"   ✓ 成功解析 {len(trajectories)} 架无人机轨迹")

        for d in data['shortest_distances']:
            assert isinstance(d['distance'], (int, float)), f"距离类型错误: {type(d['distance'])}"
            assert d['distance'] == d['distance'], f"距离为NaN"
            assert float('-inf') < d['distance'] < float('inf'), f"距离非有限: {d['distance']}"
            assert all(v == v for v in d['point_a']), f"point_a 含 NaN"
            assert all(v == v for v in d['point_b']), f"point_b 含 NaN"
        print(f"   ✓ 最短距离计算正常: {len(data['shortest_distances'])} 对")

        for o in data['overlaps']:
            assert isinstance(o['overlap_ratio'], (int, float)), f"重叠度类型错误"
            assert o['overlap_ratio'] == o['overlap_ratio'], f"重叠度为NaN"
            assert 0.0 <= o['overlap_ratio'] <= 1.0, f"重叠度范围异常: {o['overlap_ratio']}"
        print(f"   ✓ 重叠度计算正常: {len(data['overlaps'])} 对")

        print("\n2. 测试 /api/frame 接口 (重合时刻 t=5.0)...")
        r2 = requests.get(f"{API_BASE}/api/frame/5.0", timeout=10)
        assert r2.status_code == 200, f"HTTP 错误: {r2.status_code}"
        frame_data = r2.json()

        for did, pos in frame_data['drone_positions'].items():
            assert all(v == v for v in pos), f"无人机 {did} 位置含 NaN: {pos}"
            assert all(float('-inf') < v < float('inf') for v in pos), f"无人机 {did} 位置非有限"
            print(f"   ✓ {did} 位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        for d in frame_data['current_distances']:
            assert isinstance(d['distance'], (int, float)), f"距离类型错误"
            assert d['distance'] == d['distance'], f"距离为NaN"
            if d['drone_a'] in ['test_drone_A', 'test_drone_B', 'test_drone_C'] and \
               d['drone_b'] in ['test_drone_A', 'test_drone_B', 'test_drone_C']:
                assert abs(d['distance'] - 0.0) < 1e-10, f"重合时刻距离不为0: {d['distance']}"
                print(f"   ✓ {d['drone_a']} <-> {d['drone_b']}: 距离 = {d['distance']:.6f} m (完全重合 ✓)")

        print("\n3. 测试 /api/overlap-regions 接口...")
        r3 = requests.get(f"{API_BASE}/api/overlap-regions?overlap_threshold=50", timeout=10)
        assert r3.status_code == 200, f"HTTP 错误: {r3.status_code}"
        regions = r3.json()['regions']
        print(f"   ✓ 成功获取 {len(regions)} 个重叠区域")

        for region in regions:
            for p in region['points']:
                assert all(v == v for v in p['point_a']), f"重叠点含 NaN"
                assert all(v == v for v in p['point_b']), f"重叠点含 NaN"

        print(f"\n{'=' * 70}")
        print("🎉 端到端 API 测试全部通过！")
        print("   - 即使三架无人机坐标完全重合，API 也能正常返回")
        print("   - 没有 ZeroDivisionError, 没有 NaN")
        print("=" * 70)

        return True

    except AssertionError as e:
        print(f"\n✗ 断言失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n备份原始日志...")
    import shutil
    backup_dir = LOG_DIR + "_backup"
    if os.path.exists(LOG_DIR):
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(LOG_DIR, backup_dir)

    try:
        print("清空 logs 目录...")
        for f in os.listdir(LOG_DIR):
            os.remove(os.path.join(LOG_DIR, f))

        generate_collision_logs()

        success = test_api()

    finally:
        print("\n恢复原始日志...")
        for f in os.listdir(LOG_DIR):
            os.remove(os.path.join(LOG_DIR, f))
        for f in os.listdir(backup_dir):
            shutil.copy(os.path.join(backup_dir, f), os.path.join(LOG_DIR, f))
        shutil.rmtree(backup_dir)
        print("原始日志已恢复。")

    sys.exit(0 if success else 1)
