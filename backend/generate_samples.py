import os
import csv
import json
import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG_DIR = os.path.abspath(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)


def generate_spiral_trajectory(drone_id, center_x, center_y, start_z, radius, height_change, duration, steps):
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        timestamp = t * duration
        angle = t * 4 * math.pi + random.uniform(-0.1, 0.1)
        r = radius * (0.5 + 0.5 * t) + random.uniform(-5, 5)
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        z = start_z + height_change * t + random.uniform(-2, 2)
        battery = 100 - t * 80 + random.uniform(-2, 2)
        battery = max(0, min(100, battery))
        points.append({
            "timestamp": round(timestamp, 2),
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "battery": round(battery, 2)
        })
    return points


def generate_sweep_trajectory(drone_id, start_x, start_y, start_z, length_x, length_y, height_change, duration, steps):
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        timestamp = t * duration
        sweep_num = int(t * 4)
        local_t = (t * 4) - sweep_num
        if sweep_num % 2 == 0:
            x = start_x + length_x * local_t
        else:
            x = start_x + length_x * (1 - local_t)
        y = start_y + (sweep_num / 4) * length_y + random.uniform(-3, 3)
        z = start_z + height_change * t + random.uniform(-2, 2)
        battery = 100 - t * 85 + random.uniform(-2, 2)
        battery = max(0, min(100, battery))
        points.append({
            "timestamp": round(timestamp, 2),
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "battery": round(battery, 2)
        })
    return points


def generate_circular_search(drone_id, center_x, center_y, center_z, radius, duration, steps):
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        timestamp = t * duration
        angle = t * 6 * math.pi
        varying_r = radius * (1 + 0.3 * math.sin(t * 2 * math.pi))
        x = center_x + varying_r * math.cos(angle) + random.uniform(-3, 3)
        y = center_y + varying_r * math.sin(angle) + random.uniform(-3, 3)
        z = center_z + 20 * math.sin(t * 3 * math.pi) + random.uniform(-2, 2)
        battery = 100 - t * 75 + random.uniform(-2, 2)
        battery = max(0, min(100, battery))
        points.append({
            "timestamp": round(timestamp, 2),
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "battery": round(battery, 2)
        })
    return points


def save_csv(filename, points):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "x", "y", "z", "battery"])
        writer.writeheader()
        writer.writerows(points)
    print(f"Generated: {filepath}")


def save_json(filename, points):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump({"drone_id": filename.replace('.json', ''), "points": points}, f, indent=2)
    print(f"Generated: {filepath}")


if __name__ == "__main__":
    print("Generating sample drone flight logs...")

    points1 = generate_spiral_trajectory("drone_01", 100, 100, 50, 80, 30, 120, 240)
    save_csv("drone_01.csv", points1)

    points2 = generate_sweep_trajectory("drone_02", 50, 80, 60, 200, 150, -20, 120, 240)
    save_csv("drone_02.csv", points2)

    points3 = generate_circular_search("drone_03", 180, 160, 70, 60, 120, 240)
    save_json("drone_03.json", points3)

    points4 = generate_spiral_trajectory("drone_04", 220, 80, 45, 70, 40, 100, 200)
    save_csv("drone_04.csv", points4)

    print("\nDone! Generated 4 sample log files in:", LOG_DIR)
    print("Files: drone_01.csv, drone_02.csv, drone_03.json, drone_04.csv")
