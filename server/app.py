"""
Traffic Signal Control — FastAPI Server
Uses openenv-core create_fastapi_app pattern.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import TrafficAction, TrafficObservation, TrafficState
from server.traffic_environment import TrafficSignalEnvironment

# ── Global environment instance ───────────────────────────────────────────────
_env = TrafficSignalEnvironment()

# ── Try to use openenv-core create_fastapi_app ────────────────────────────────
try:
    from openenv.core.env_server import create_fastapi_app
    app = create_fastapi_app(_env, TrafficAction, TrafficObservation)
except Exception:
    # Fallback: manual FastAPI app
    app = FastAPI(
        title="Traffic Signal Control - OpenEnv",
        description="Real-world traffic signal control environment for RL agents.",
        version="1.0.0",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "single_intersection_easy"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    phase_assignments: dict = {}
    duration: int = 5
    task_id: Optional[str] = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "traffic-signal-env"}

@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    obs = _env.reset(task_id=request.task_id, seed=request.seed)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {"task_id": request.task_id},
    }

@app.post("/step")
async def step(request: StepRequest):
    action = TrafficAction(
        phase_assignments=request.phase_assignments,
        duration=request.duration,
        task_id=request.task_id or _env._task_id,
    )
    obs = _env.step(action)
    info = {
        "step_number": obs.step_number,
        "network_throughput": obs.network_throughput,
        "network_avg_wait_time": obs.network_avg_wait_time,
    }
    if obs.done and "final_score" in obs.metadata:
        info["final_score"] = obs.metadata["final_score"]
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": info,
    }

@app.get("/state")
async def state():
    return _env.state.model_dump()

@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {"id": "single_intersection_easy", "difficulty": "easy",  "max_steps": 100},
            {"id": "arterial_corridor_medium", "difficulty": "medium", "max_steps": 150},
            {"id": "urban_grid_hard",          "difficulty": "hard",   "max_steps": 200},
        ]
    }

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
