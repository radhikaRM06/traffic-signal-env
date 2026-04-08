#!/usr/bin/env python3
"""
Traffic Signal Control — OpenEnv Inference Script
Emits structured [START] / [STEP] / [END] logs.
"""

import json
import os
import sys
import time
import requests

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL  = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN      = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
TRAFFIC_HOST  = os.environ.get("TRAFFIC_HOST", "https://radhikamishra-traffic-signal-env.hf.space")

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

# ─── Structured Logging (text format) ────────────────────────────────────────

def log_start(task_id):
    print(f"[START] task={task_id} model={MODEL_NAME}", flush=True)

def log_step(task_id, step, reward, done, action, info):
    print(f"[STEP] task={task_id} step={step} reward={round(reward,4)} done={done} action={json.dumps(action)} info={json.dumps(info)}", flush=True)

def log_end(task_id, total_reward, final_score, steps, elapsed):
    print(f"[END] task={task_id} score={round(final_score,4)} total_reward={round(total_reward,4)} steps={steps} elapsed={round(elapsed,2)}", flush=True)

# ─── Environment HTTP calls ───────────────────────────────────────────────────

def env_reset(task_id):
    try:
        r = requests.post(
            f"{TRAFFIC_HOST}/reset",
            json={"task_id": task_id},
            timeout=60
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
            timeout=60
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] step failed: {e}", flush=True)
        return None

# ─── Heuristic agent ─────────────────────────────────────────────────────────

def heuristic_action(obs_result):
    try:
        intersections = obs_result.get("observation", {}).get("intersections", [])
        phases = {}
        for i in intersections:
            iid = i["intersection_id"]
            if i.get("emergency_vehicle_present"):
                d = i.get("emergency_vehicle_direction", "N")
                phases[iid] = 0 if d in ["N", "S"] else 2
            elif i.get("pedestrian_demand"):
                phases[iid] = 4
            else:
                ns = i["queue_north"] + i["queue_south"]
                ew = i["queue_east"] + i["queue_west"]
                phases[iid] = 0 if ns >= ew else 2
        return phases
    except Exception:
        return {}

# ─── LLM agent ───────────────────────────────────────────────────────────────

def get_llm_action(obs_result, task_id):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        intersections = obs_result.get("observation", {}).get("intersections", [])
        lines = [f"Task: {task_id}"]
        for i in intersections:
            lines.append(
                f"Intersection {i['intersection_id']}: "
                f"phase={i['current_phase']} "
                f"queues N={i['queue_north']} S={i['queue_south']} "
                f"E={i['queue_east']} W={i['queue_west']} "
                f"emergency={i['emergency_vehicle_present']} "
                f"ped={i['pedestrian_demand']}"
            )

        system = (
            "You are a traffic signal controller. "
            "Return ONLY a JSON object like {\"phase_assignments\": {\"I0\": 0}}. "
            "Phase 0=NS_GREEN, 2=EW_GREEN, 4=ALL_RED. "
            "Give green to direction with longer queue."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": "\n".join(lines)},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("phase_assignments", heuristic_action(obs_result))
    except Exception as e:
        print(f"[WARN] LLM action failed ({e}), using heuristic", flush=True)
        return heuristic_action(obs_result)

# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(task_id):
    max_steps    = TASK_MAX_STEPS[task_id]
    use_llm      = bool(HF_TOKEN)
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
            if use_llm:
                phase_assignments = get_llm_action(obs_result, task_id)
            else:
                phase_assignments = heuristic_action(obs_result)

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

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Traffic Signal Control — OpenEnv Baseline", flush=True)
    print(f"[INFO] Model={MODEL_NAME} Host={TRAFFIC_HOST}", flush=True)

    # Wait for environment to be ready
    for attempt in range(10):
        try:
            r = requests.get(f"{TRAFFIC_HOST}/health", timeout=30)
            if r.status_code == 200:
                print(f"[INFO] Environment healthy: {r.json()}", flush=True)
                break
        except Exception as e:
            print(f"[WAIT] attempt {attempt+1}/10: {e}", flush=True)
            time.sleep(10)
    else:
        print("[ERROR] Environment not reachable after 10 attempts", flush=True)
        sys.exit(1)

    results = []
    for task_id in TASKS:
        print(f"[INFO] Starting task: {task_id}", flush=True)
        try:
            result = run_episode(task_id)
            results.append(result)
            print(f"[INFO] Completed {task_id} score={result['final_score']:.4f}", flush=True)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
            results.append({"task_id": task_id, "final_score": 0.0,
                            "total_reward": 0.0, "steps": 0})
            log_end(task_id, 0.0, 0.0, 0, 0.0)

    avg = sum(r["final_score"] for r in results) / max(1, len(results))
    print(f"[SUMMARY] average_score={round(avg,4)} model={MODEL_NAME}", flush=True)


if __name__ == "__main__":
    main()
