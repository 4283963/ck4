import os
import sys
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    DroneTrajectory, ShortestDistanceResult, OverlapResult,
    AnalysisResponse, FrameData
)
from log_parser import (
    load_logs_from_dir, all_trajectories_to_numpy,
    parse_csv_log, parse_json_log
)
from calculator import (
    compute_all_shortest_distances, compute_all_overlaps,
    get_time_range, interpolate_position, get_overlap_region_points
)
import numpy as np

app = FastAPI(title="无人机搜救轨迹分析系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG_DIR = os.path.abspath(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "无人机搜救轨迹分析系统 API 运行中"}


@app.post("/api/upload-log")
async def upload_log(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_path = os.path.join(LOG_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        return {"filename": file.filename, "size": len(content), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
async def list_logs():
    if not os.path.exists(LOG_DIR):
        return {"files": []}
    files = [f for f in sorted(os.listdir(LOG_DIR)) if f.endswith(('.csv', '.json'))]
    return {"files": files}


@app.get("/api/analyze", response_model=AnalysisResponse)
async def analyze_logs(overlap_threshold: Optional[float] = 50.0):
    trajectories = load_logs_from_dir(LOG_DIR)
    if not trajectories:
        raise HTTPException(status_code=404, detail="未找到日志文件")

    traj_np = all_trajectories_to_numpy(trajectories)
    shortest_distances = compute_all_shortest_distances(traj_np)
    overlaps = compute_all_overlaps(traj_np, threshold=overlap_threshold)
    t_min, t_max = get_time_range(traj_np)

    return AnalysisResponse(
        trajectories=trajectories,
        shortest_distances=shortest_distances,
        overlaps=overlaps,
        time_range=[t_min, t_max]
    )


@app.get("/api/frame/{timestamp}", response_model=FrameData)
async def get_frame(timestamp: float, overlap_threshold: Optional[float] = 50.0):
    trajectories = load_logs_from_dir(LOG_DIR)
    if not trajectories:
        raise HTTPException(status_code=404, detail="未找到日志文件")

    traj_np = all_trajectories_to_numpy(trajectories)
    drone_positions = {}
    for did, traj in traj_np.items():
        pos = interpolate_position(traj, timestamp)
        drone_positions[did] = pos.tolist()

    current_distances = []
    ids = list(traj_np.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pa = interpolate_position(traj_np[ids[i]], timestamp)
            pb = interpolate_position(traj_np[ids[j]], timestamp)
            dist = float(np.sqrt(np.sum((pa - pb) ** 2)))
            current_distances.append(ShortestDistanceResult(
                drone_a=ids[i], drone_b=ids[j],
                distance=dist, timestamp=timestamp,
                point_a=pa.tolist(), point_b=pb.tolist()
            ))

    return FrameData(
        timestamp=timestamp,
        drone_positions=drone_positions,
        current_distances=current_distances
    )


@app.get("/api/overlap-regions")
async def get_overlap_regions(overlap_threshold: Optional[float] = 50.0):
    trajectories = load_logs_from_dir(LOG_DIR)
    if not trajectories:
        return {"regions": []}

    traj_np = all_trajectories_to_numpy(trajectories)
    ids = list(traj_np.keys())
    regions = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            points = get_overlap_region_points(
                traj_np[ids[i]], traj_np[ids[j]],
                threshold=overlap_threshold
            )
            if points:
                region_points = []
                for pa, pb, t in points:
                    mid = ((pa + pb) / 2).tolist()
                    region_points.append({
                        "timestamp": t,
                        "point_a": pa.tolist(),
                        "point_b": pb.tolist(),
                        "midpoint": mid
                    })
                regions.append({
                    "drone_a": ids[i],
                    "drone_b": ids[j],
                    "points": region_points
                })
    return {"regions": regions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
