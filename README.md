---
<<<<<<< HEAD
title: Traffic Signal Control — OpenEnv
emoji: 🚦
colorFrom: green
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - urban-mobility
  - real-world
---

# 🚦 Traffic Signal Control — OpenEnv Environment

A **real-world reinforcement learning environment** where an AI agent learns to manage urban intersection traffic signals to minimize vehicle wait times and maximize throughput.

---

## 🌍 Why This Environment?

Traffic signal control is one of the most impactful real-world applications of RL:
- **Urban mobility**: Poorly timed signals waste millions of vehicle-hours daily
- **Emissions**: Idling vehicles are a major source of city CO₂
- **Safety**: Proper signaling reduces accident risk
- **Emergency response**: Signal preemption saves lives

Unlike toy environments, this directly mirrors the decision problem faced by adaptive traffic control systems (ATCS) deployed in real cities.

---

## 🏗️ Environment Architecture

```
┌────────────────────────────────────────────────────────────┐
│  AGENT (your training code / inference.py)                 │
│  env_client.reset() → env_client.step(action)             │
└─────────────────────┬──────────────────────────────────────┘
                      │ HTTP JSON  POST /reset  POST /step  GET /state
┌─────────────────────▼──────────────────────────────────────┐
│  FastAPI Server (Docker / HF Space)                        │
│  ├── TrafficSignalEnvironment                              │
│  │   ├── TrafficNetwork (simulation engine)               │
│  │   │   └── Intersection × N (discrete-event sim)       │
│  │   └── Graders (score 0.0–1.0)                         │
│  └── Endpoints: /reset  /step  /state  /health  /tasks    │
└────────────────────────────────────────────────────────────┘
```

---

## 🎮 Action Space

```python
class TrafficAction(BaseModel):
    phase_assignments: Dict[str, int]  # {intersection_id: phase_id}
    duration: int = 5                  # ticks to hold phase (1-10)
    task_id: str                       # which task to run
```

### Phase IDs

| Phase | Name | NS Signal | EW Signal | Use Case |
|-------|------|-----------|-----------|----------|
| 0 | NS_GREEN | 🟢 GREEN | 🔴 RED | N/S traffic flows |
| 1 | NS_YELLOW | 🟡 YELLOW | 🔴 RED | N/S transition |
| 2 | EW_GREEN | 🔴 RED | 🟢 GREEN | E/W traffic flows |
| 3 | EW_YELLOW | 🔴 RED | 🟡 YELLOW | E/W transition |
| 4 | ALL_RED | 🔴 RED | 🔴 RED | Pedestrian crossing |

---

## 👁️ Observation Space

```python
class TrafficObservation(BaseModel):
    # Per-intersection
    intersections: List[IntersectionObservation]
    # Each intersection contains:
    #   current_phase, phase_time_elapsed
    #   queue_north/south/east/west  (vehicle counts)
    #   avg_wait_north/south/east/west  (seconds)
    #   throughput_last_tick
    #   emergency_vehicle_present, emergency_vehicle_direction
    #   pedestrian_demand

    # Network-wide
    total_vehicles_waiting: int
    network_avg_wait_time: float
    network_throughput: int
    time_of_day: str           # morning/afternoon/evening/night
    sim_time: int
    done: bool
    reward: float
```

---

## 📋 Tasks

### Task 1: Single Intersection (Easy) 🟢
- **Intersections**: 1
- **Demand**: Moderate (~0.5 vehicles/tick/approach)
- **Max Steps**: 100
- **Grader** (score 0–1):
  - 60% average wait time reduction
  - 25% total throughput
  - 15% emergency clearance rate
- **Expected baseline score**: ~0.45 (random) → ~0.75 (optimal heuristic)

### Task 2: Arterial Corridor (Medium) 🟡
- **Intersections**: 3 (sequential: I0 → I1 → I2)
- **Demand**: High (~0.65 vehicles/tick/approach)
- **Max Steps**: 150
- **Grader** (score 0–1):
  - 50% network average wait time
  - 30% total throughput
  - 20% green-wave coordination (all intersections on same axis)
- **Expected baseline score**: ~0.35 (random) → ~0.65 (coordinated heuristic)

### Task 3: Urban Grid (Hard) 🔴
- **Intersections**: 4 (2×2 grid: I00, I01, I10, I11)
- **Demand**: Rush-hour (~0.85 vehicles/tick/approach)
- **Max Steps**: 200
- **Grader** (score 0–1):
  - 40% network wait time
  - 25% throughput
  - 20% emergency vehicle clearance speed
  - 10% pedestrian phase utilization
  - 5% phase stability (penalty for oscillation)
- **Expected baseline score**: ~0.25 (random) → ~0.55 (optimized heuristic)

---

## 🏆 Reward Function

The reward is shaped to provide dense, informative signal at every step:

```
reward = throughput_reward - wait_penalty - emergency_penalty + pedestrian_bonus

where:
  throughput_reward  = vehicles_passed_this_tick × 0.3
  wait_penalty       = (total_wait / max_possible_wait) × 2.0
  emergency_penalty  = 0.5  if any emergency vehicle not yet cleared
  pedestrian_bonus   = 0.1  if ALL_RED given when pedestrian demand present

reward ∈ [-2.0, 2.0]  (clipped)
```

**Design rationale:**
- Dense signal at every step (not sparse end-of-episode)
- Throughput reward encourages green phases for high-queue directions
- Wait penalty discourages leaving vehicles idle
- Emergency penalty creates urgency for preemption
- Pedestrian bonus teaches the agent to serve ALL_RED phases appropriately

---

## 🚀 Setup & Usage

### Docker (Recommended)

```bash
# Build
docker build -t traffic-signal-env .

# Run
docker run -p 8000:7860 \
  -e TRAFFIC_TASK=single_intersection_easy \
  -e TRAFFIC_SEED=42 \
  traffic-signal-env
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start server (from project root)
cd src
TRAFFIC_TASK=single_intersection_easy PYTHONPATH=. \
  uvicorn envs.traffic_signal_env.server.app:app --port 8000 --reload
```

### API Usage

```python
import requests

BASE = "http://localhost:8000"

# Reset
obs = requests.post(f"{BASE}/reset", json={"task_id": "single_intersection_easy"}).json()

# Step
result = requests.post(f"{BASE}/step", json={
    "phase_assignments": {"I0": 0},  # NS GREEN
    "task_id": "single_intersection_easy"
}).json()

print(result["reward"])   # e.g. 0.45
print(result["done"])     # False until max_steps
print(result["info"]["final_score"])  # 0.0–1.0 when done=True

# State
state = requests.get(f"{BASE}/state").json()
print(state["step_count"], state["total_throughput"])
```

### Run Baseline Inference

```bash
# With LLM agent
export HF_TOKEN=your_api_key
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
export TRAFFIC_HOST=http://localhost:8000
python inference.py

# Heuristic baseline (no API key needed)
TRAFFIC_HOST=http://localhost:8000 python inference.py
```

---

## 📊 Baseline Scores

Heuristic agent (queue-length-based, no learning):

| Task | Score | Notes |
|------|-------|-------|
| single_intersection_easy | ~0.62 | Consistent green-wave for long queues |
| arterial_corridor_medium | ~0.48 | No coordination, but reactive |
| urban_grid_hard | ~0.38 | Struggles with 4-way coordination |

GPT-4o-mini agent:

| Task | Score | Notes |
|------|-------|-------|
| single_intersection_easy | ~0.71 | Understands emergency preemption |
| arterial_corridor_medium | ~0.57 | Some green-wave coordination |
| urban_grid_hard | ~0.44 | Handles pedestrian phases well |

---

## 📁 Project Structure

```
traffic-signal-env/
├── openenv.yaml                          # OpenEnv metadata
├── Dockerfile                            # Container definition
├── requirements.txt
├── inference.py                          # Baseline inference script ← ROOT
├── README.md
└── src/
    ├── core/
    │   └── __init__.py                   # Base Environment/Action/Observation
    └── envs/
        └── traffic_signal_env/
            ├── __init__.py
            ├── models.py                 # Typed Pydantic models
            ├── simulation.py             # Discrete-event traffic engine
            ├── environment.py            # OpenEnv Environment implementation
            ├── graders.py                # Task graders (0.0–1.0 scoring)
            └── server/
                ├── __init__.py
                └── app.py                # FastAPI HTTP server
```

---

## 🔬 How the Simulation Works

Each intersection runs a discrete-event simulation:

1. **Arrivals**: Vehicles arrive via Poisson process (configurable rate per approach)
2. **Serving**: Green approaches discharge up to 3 vehicles/tick (saturation flow)
3. **Wait accumulation**: Queued vehicles accumulate wait time each tick
4. **Events**: Emergency vehicles (1%/tick) and pedestrian demand (10%/tick) spawn randomly
5. **Demand scaling**: Rush-hour multiplies arrival rates by 1.8×

The simulation is calibrated against real-world traffic engineering parameters (HCM saturation flow, typical urban arrival rates).

---

## 📝 License

MIT License
=======
title: Traffic Signal Env
emoji: 👁
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: 'AI-based Traffic signal simulation environment '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 86be67a5c79623d401a0dfdb9926cc33a2f2ca9a
