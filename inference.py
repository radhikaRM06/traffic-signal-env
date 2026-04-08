#!/usr/bin/env python3
"""
Traffic Signal Control — OpenEnv Inference Script

Runs an LLM-based agent against all 3 tasks and reports scores.
Uses OpenAI client with configurable API_BASE_URL, MODEL_NAME, HF_TOKEN.

Emits structured stdout logs in [START] / [STEP] / [END] format.

Usage:
    python inference.py

Environment Variables:
    API_BASE_URL   - LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     - Model identifier (default: gpt-4o-mini)
    HF_TOKEN       - API key (also accepts OPENAI_API_KEY)
    TRAFFIC_HOST   - Environment host (default: http://localhost:8000)
"""

import json
import os
import sys
import time
import random
import requests
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
TRAFFIC_HOST = os.environ.get("TRAFFIC_HOST", "http://localhost:8000")

TASKS = [
    "single_intersection_easy",
    "arterial_corridor_medium",
    "urban_grid_hard",
]

TASK_MAX_STEPS = {
    "single_intersection_easy": 100,
    "arterial_corridor_medium": 150,
    "urban_grid_hard": 200,
}

# ─── OpenAI Client ────────────────────────────────────────────────────────────

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ─── Environment Client ───────────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{TRAFFIC_HOST}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(phase_assignments: Dict[str, int], task_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{TRAFFIC_HOST}/step",
        json={"phase_assignments": phase_assignments, "task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

def env_state() -> Dict[str, Any]:
    resp = requests.get(f"{TRAFFIC_HOST}/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─── LLM Agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert traffic signal controller.

Your goal is to minimize vehicle wait times and maximize throughput at urban intersections.

At each step you receive observations about queue lengths and current signal phases.
You must return a JSON object with phase_assignments mapping each intersection_id to a phase_id.

Phase IDs:
  0 = North-South GREEN  (vehicles on N/S can go)
  1 = North-South YELLOW (transition — brief)
  2 = East-West  GREEN   (vehicles on E/W can go)
  3 = East-West  YELLOW  (transition — brief)
  4 = ALL RED            (pedestrian crossing only)

Strategy guidelines:
- Give green to the direction with longer queues
- Use phase 4 (ALL RED) when pedestrian_demand is True
- If emergency_vehicle_present, immediately give green to emergency_vehicle_direction axis
- Avoid switching phases every tick (causes oscillation penalty)
- For arterial corridors, try to coordinate all intersections on same axis (green wave)

Respond ONLY with valid JSON like:
{"phase_assignments": {"I0": 0, "I1": 2}}

No explanation, no markdown, just the JSON.
"""

def build_user_prompt(obs: Dict[str, Any]) -> str:
    intersections = obs.get("observation", {}).get("intersections", [])
    lines = [
        f"Step {obs.get('observation', {}).get('step_number', '?')} | "
        f"Time: {obs.get('observation', {}).get('time_of_day', '?')} | "
        f"Total waiting: {obs.get('observation', {}).get('total_vehicles_waiting', 0)}"
    ]
    for i in intersections:
        lines.append(
            f"\nIntersection {i['intersection_id']}:"
            f"\n  Phase: {i['current_phase']} (held {i['phase_time_elapsed']} ticks)"
            f"\n  Queues — N:{i['queue_north']} S:{i['queue_south']} "
            f"E:{i['queue_east']} W:{i['queue_west']}"
            f"\n  Emergency: {i['emergency_vehicle_present']} "
            f"({i.get('emergency_vehicle_direction', 'none')})"
            f"\n  Pedestrian demand: {i['pedestrian_demand']}"
        )
    return "\n".join(lines)


def get_llm_action(obs_result: Dict[str, Any], task_id: str) -> Dict[str, int]:
    """Query LLM for phase assignments. Falls back to heuristic on error."""
    try:
        user_msg = build_user_prompt(obs_result)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("phase_assignments", {})
    except Exception as e:
        # Fallback: heuristic
        return heuristic_action(obs_result)


def heuristic_action(obs_result: Dict[str, Any]) -> Dict[str, int]:
    """
    Simple heuristic: give green to direction with most vehicles waiting.
    Emergency preemption included.
    """
    intersections = obs_result.get("observation", {}).get("intersections", [])
    phases = {}
    for i in intersections:
        iid = i["intersection_id"]

        # Emergency preemption
        if i.get("emergency_vehicle_present"):
            direction = i.get("emergency_vehicle_direction", "N")
            phases[iid] = 0 if direction in ["N", "S"] else 2
            continue

        # Pedestrian phase
        if i.get("pedestrian_demand"):
            phases[iid] = 4
            continue

        # Longest queue wins
        ns_queue = i["queue_north"] + i["queue_south"]
        ew_queue = i["queue_east"] + i["queue_west"]
        phases[iid] = 0 if ns_queue >= ew_queue else 2

    return phases


# ─── Structured Logging ───────────────────────────────────────────────────────

def log_start(task_id: str, model: str):
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "model": model,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()

def log_step(task_id: str, step: int, reward: float, done: bool,
             action: Dict, info: Dict):
    print(json.dumps({
        "type": "STEP",
        "task_id": task_id,
        "step": step,
        "reward": reward,
        "done": done,
        "action": action,
        "info": info,
    }))
    sys.stdout.flush()

def log_end(task_id: str, total_reward: float, final_score: float,
            steps: int, elapsed: float):
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "final_score": round(final_score, 4),
        "steps": steps,
        "elapsed_seconds": round(elapsed, 2),
    }))
    sys.stdout.flush()


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(task_id: str, use_llm: bool = True) -> Dict[str, Any]:
    """Run one complete episode and return results."""
    max_steps = TASK_MAX_STEPS[task_id]
    t0 = time.time()

    log_start(task_id, MODEL_NAME)

    # Reset
    obs_result = env_reset(task_id)
    total_reward = 0.0
    final_score = 0.0
    step = 0

    for step in range(1, max_steps + 1):
        # Get action
        if use_llm and HF_TOKEN:
            phase_assignments = get_llm_action(obs_result, task_id)
        else:
            phase_assignments = heuristic_action(obs_result)

        # Step
        result = env_step(phase_assignments, task_id)
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        info = result.get("info", {})
        total_reward += reward

        log_step(task_id, step, reward, done, phase_assignments, info)

        obs_result = result

        if done:
            final_score = info.get("final_score", 0.0)
            break

    elapsed = time.time() - t0
    log_end(task_id, total_reward, final_score, step, elapsed)

    return {
        "task_id": task_id,
        "total_reward": round(total_reward, 4),
        "final_score": final_score,
        "steps": step,
        "elapsed": round(elapsed, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Traffic Signal Control — OpenEnv Baseline")
    print(f"Model : {MODEL_NAME}")
    print(f"Host  : {TRAFFIC_HOST}")
    print(f"Tasks : {TASKS}")
    print("=" * 60)
    print()

    # Check environment is up
    try:
        resp = requests.get(f"{TRAFFIC_HOST}/health", timeout=10)
        resp.raise_for_status()
        print(f"[OK] Environment health: {resp.json()}\n")
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {TRAFFIC_HOST}: {e}")
        print("  Start the server: cd src && uvicorn envs.traffic_signal_env.server.app:app --port 8000")
        sys.exit(1)

    use_llm = bool(HF_TOKEN)
    if not use_llm:
        print("[WARN] No HF_TOKEN/OPENAI_API_KEY found. Using heuristic baseline.\n")

    results = []
    for task_id in TASKS:
        print(f"\n{'─'*60}")
        print(f"Running task: {task_id}")
        print(f"{'─'*60}")
        try:
            result = run_episode(task_id, use_llm=use_llm)
            results.append(result)
            print(f"\n  ✓ Score: {result['final_score']:.4f} | "
                  f"Reward: {result['total_reward']:.2f} | "
                  f"Steps: {result['steps']}")
        except Exception as e:
            print(f"  ✗ Task failed: {e}")
            results.append({"task_id": task_id, "final_score": 0.0, "error": str(e)})

    print(f"\n{'='*60}")
    print("FINAL SCORES")
    print(f"{'='*60}")
    for r in results:
        score = r.get("final_score", 0.0)
        bar = "█" * int(score * 30)
        print(f"  {r['task_id']:35s} [{bar:<30}] {score:.4f}")

    avg_score = sum(r.get("final_score", 0.0) for r in results) / max(1, len(results))
    print(f"\n  Average score: {avg_score:.4f}")
    print(f"{'='*60}\n")

    # Machine-readable summary
    print(json.dumps({
        "type": "SUMMARY",
        "results": results,
        "average_score": round(avg_score, 4),
        "model": MODEL_NAME,
    }))


if __name__ == "__main__":
    main()
