"""
Traffic Signal Control - FastAPI Server

Exposes the OpenEnv standard HTTP API:
  POST /reset    → returns initial observation
  POST /step     → returns (observation, reward, done, info)
  GET  /state    → returns current episode state
  GET  /health   → liveness check
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from envs.traffic_signal_env.environment import TrafficSignalEnvironment
from envs.traffic_signal_env.models import TrafficAction


# ─── Global env instance (one per container) ──────────────────────────────────

TASK_ID = os.environ.get("TRAFFIC_TASK", "single_intersection_easy")
SEED = int(os.environ.get("TRAFFIC_SEED", "42"))

env = TrafficSignalEnvironment(task_id=TASK_ID, seed=SEED)
_last_obs: Dict[str, Any] = {}
_total_reward: float = 0.0


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Traffic Signal Control - OpenEnv",
    description=(
        "Real-world traffic signal control environment. "
        "An AI agent manages urban intersection signal phases to minimize "
        "vehicle wait times and maximize throughput."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    phase_assignments: Dict[str, int]
    duration: int = 5
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok", "environment": "traffic_signal_control", "task": TASK_ID}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """Start a new episode. Returns initial observation."""
    global env, _last_obs, _total_reward

    task_id = request.task_id or TASK_ID
    seed = request.seed if request.seed is not None else SEED

    if task_id != env.task_id or seed != env.seed:
        env = TrafficSignalEnvironment(task_id=task_id, seed=seed)

    obs = env.reset()
    _last_obs = obs
    _total_reward = 0.0

    return {
        "observation": obs,
        "reward": 0.0,
        "done": False,
        "info": {"task_id": task_id, "message": "Episode started"},
    }


@app.post("/step")
async def step(request: StepRequest):
    """Execute one action. Returns observation, reward, done, info."""
    global _last_obs, _total_reward

    if env.network is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    action = {
        "phase_assignments": request.phase_assignments,
        "duration": request.duration,
        "task_id": request.task_id or env.task_id,
        "metadata": request.metadata,
    }

    obs = env.step(action)
    reward = obs.get("reward", 0.0)
    done = obs.get("done", False)
    _last_obs = obs
    _total_reward += reward

    info = {
        "step_number": obs.get("step_number", 0),
        "sim_time": obs.get("sim_time", 0),
        "total_reward_so_far": round(_total_reward, 4),
        "network_throughput": obs.get("network_throughput", 0),
        "network_avg_wait_time": obs.get("network_avg_wait_time", 0.0),
    }

    if done and "final_score" in obs.get("metadata", {}):
        info["final_score"] = obs["metadata"]["final_score"]
        info["episode_stats"] = obs["metadata"].get("episode_stats", {})

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state():
    """Return current episode state/metadata."""
    return env.get_state()


@app.get("/tasks")
async def list_tasks():
    """List available tasks."""
    return {
        "tasks": [
            {
                "id": "single_intersection_easy",
                "name": "Single Intersection Control",
                "difficulty": "easy",
                "max_steps": 100,
                "description": "Control 1 intersection with moderate traffic demand.",
            },
            {
                "id": "arterial_corridor_medium",
                "name": "Arterial Corridor Optimization",
                "difficulty": "medium",
                "max_steps": 150,
                "description": "Coordinate 3 sequential intersections for green-wave efficiency.",
            },
            {
                "id": "urban_grid_hard",
                "name": "Urban Grid Network Control",
                "difficulty": "hard",
                "max_steps": 200,
                "description": "Manage a 2×2 grid under rush-hour demand with emergencies.",
            },
        ]
    }
