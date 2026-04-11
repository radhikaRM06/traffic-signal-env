#!/usr/bin/env python3
"""
Traffic Signal Control — OpenEnv Inference Script

Required env vars:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       API key
    TRAFFIC_HOST   Environment URL (default: HF Space URL)
"""

import json
import os
import sys
import time
import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
TRAFFIC_HOST = os.environ.get(
    "TRAFFIC_HOST",
    "https://radhikamishra-traffic-signal-env.hf.space"
)

TASKS = [
    "single_intersection_easy",
    "arterial_corridor_medium",
    "urban_grid_hard",
]

TASK_MAX_STEPS = {
    "single_intersection_easy": 100,
    "arterial_corridor_medium": 150,
    "urban_grid_hard":          200,
}

# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task_id):
    print(f"[START] task={task_id} model={MODEL_NAME}", flush=True)

def log_step(task_id, step, reward, done, action, info):
    print(
        f"[STEP] task={task_id} step={step} "
        f"reward={round(float(reward), 4)} done={done} "
        f"action={json.dumps(action)} info={json.dumps(info)}",
        flush=True
    )

def log_end(task_id, total_reward, final_score, steps, elapsed):
    print(
        f"[END] task={task_id} score={round(float(final_score), 4)} "
        f"total_reward={round(float(total_reward), 4)} "
        f"steps={steps} elapsed={round(elapsed, 2)}",
        flush=True
    )

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def env_reset(task_id):
    try:
        r = requests.post(
            f"{TRAFFIC_HOST}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] reset failed: {e}", flush=True)
        return None

def env_step(phase_assignments, task_id):
    try:
        r = requests.post(
            f"{TRAFFIC_HOST}/step",
            json={"phase_assignments": phase_assignments, "task_id": task_id},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] step failed: {e}", flush=True)
        return None

# ── Heuristic agent ───────────────────────────────────────────────────────────

def heuristic_action(obs_result):
    try:
        obs = obs_result.get("observation", obs_result)
        intersections = obs.get("intersections", [])
        phases = {}
        for i in intersections:
            iid = i.get("intersection_id", "I0")
            if i.get("emergency_vehicle_present"):
                d = i.get("emergency_vehicle_direction", "N")
                phases[iid] = 0 if d in ["N", "S"] else 2
            elif i.get("pedestrian_demand"):
                phases[iid] = 4
            else:
                ns = i.get("queue_north", 0) + i.get("queue_south", 0)
                ew = i.get("queue_east", 0) + i.get("queue_west", 0)
                phases[iid] = 0 if ns >= ew else 2
        if not phases:
            phases = {"I0": 0}
        return phases
    except Exception:
        return {"I0": 0}

# ── LLM agent ─────────────────────────────────────────────────────────────────

def get_llm_action(obs_result, task_id):
    if not HF_TOKEN:
        return heuristic_action(obs_result)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        obs = obs_result.get("observation", obs_result)
        intersections = obs.get("intersections", [])
        lines = [f"Task: {task_id}. Control traffic signals to minimize wait times."]
        for i in intersections:
            lines.append(
                f"Intersection {i.get('intersection_id','I0')}: "
                f"phase={i.get('current_phase',0)} "
                f"N={i.get('queue_north',0)} S={i.get('queue_south',0)} "
                f"E={i.get('queue_east',0)} W={i.get('queue_west',0)} "
                f"emergency={i.get('emergency_vehicle_present',False)} "
                f"emergency_dir={i.get('emergency_vehicle_direction','none')} "
                f"pedestrian={i.get('pedestrian_demand',False)}"
            )
        system = (
            "You are a traffic signal controller. "
            "Return ONLY valid JSON: {\"phase_assignments\": {\"I0\": 0}}. "
            "Phases: 0=NS_GREEN 2=EW_GREEN 4=ALL_RED. "
            "Rules: give green to longer queue; phase 4 if pedestrian; "
            "match emergency_dir axis immediately."
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(lines)},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        result = parsed.get("phase_assignments", {})
        return result if result else heuristic_action(obs_result)
    except Exception as e:
        print(f"[WARN] LLM failed ({e}), using heuristic", flush=True)
        return heuristic_action(obs_result)

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id):
    max_steps    = TASK_MAX_STEPS.get(task_id, 100)
    t0           = time.time()
    total_reward = 0.0
    final_score  = 0.0
    step         = 0

    log_start(task_id)

    obs_result = env_reset(task_id)
    if obs_result is None:
        log_end(task_id, 0.0, 0.0, 0, time.time() - t0)
        return {"task_id": task_id, "final_score": 0.0, "total_reward": 0.0, "steps": 0}

    for step in range(1, max_steps + 1):
        try:
            phase_assignments = get_llm_action(obs_result, task_id)
            if not phase_assignments:
                phase_assignments = heuristic_action(obs_result)

            result = env_step(phase_assignments, task_id)
            if result is None:
                break

            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))
            info   = result.get("info", {})
            total_reward += reward

            log_step(task_id, step, reward, done, phase_assignments, info)
            obs_result = result

            if done:
                final_score = float(info.get("final_score", 0.0))
                break

        except Exception as e:
            print(f"[WARN] step {step} error: {e}", flush=True)
            continue

    elapsed = time.time() - t0
    log_end(task_id, total_reward, final_score, step, elapsed)

    return {
        "task_id":      task_id,
        "final_score":  final_score,
        "total_reward": round(total_reward, 4),
        "steps":        step,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Traffic Signal Control — OpenEnv Inference", flush=True)
    print(f"[INFO] model={MODEL_NAME} host={TRAFFIC_HOST}", flush=True)

    # Wait for environment to be ready (up to 5 mins)
    healthy = False
    for attempt in range(30):
        try:
            r = requests.get(f"{TRAFFIC_HOST}/health", timeout=30)
            if r.status_code == 200:
                print(f"[INFO] Environment ready: {r.json()}", flush=True)
                healthy = True
                break
        except Exception as e:
            print(f"[WAIT] {attempt+1}/30 ({e})", flush=True)
        time.sleep(10)

    if not healthy:
        print("[ERROR] Environment not reachable", flush=True)
        sys.exit(1)

    results = []
    for task_id in TASKS:
        print(f"[INFO] === Task: {task_id} ===", flush=True)
        try:
            result = run_episode(task_id)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_id} crashed: {e}", flush=True)
            log_end(task_id, 0.0, 0.0, 0, 0.0)
            results.append({
                "task_id": task_id,
                "final_score": 0.0,
                "total_reward": 0.0,
                "steps": 0,
            })

    avg = sum(r["final_score"] for r in results) / max(1, len(results))
    print(f"[SUMMARY] average_score={round(avg,4)} model={MODEL_NAME}", flush=True)
    for r in results:
        print(
            f"[RESULT] task={r['task_id']} "
            f"score={r['final_score']} "
            f"reward={r['total_reward']} "
            f"steps={r['steps']}",
            flush=True
        )


if __name__ == "__main__":
    main()
