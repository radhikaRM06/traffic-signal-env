---
title: Traffic Signal Control OpenEnv
emoji: 🚦
colorFrom: green
colorTo: red
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - urban-mobility
  - real-world
short_description: 'Real-world traffic signal control environment for RL agents'
---

# Traffic Signal Control - OpenEnv

A real-world reinforcement learning environment where an AI agent learns to manage urban intersection traffic signals to minimize vehicle wait times and maximize throughput.

## Why This Environment?

Traffic signal control is one of the most impactful real-world applications of RL. Poorly timed signals waste millions of vehicle-hours daily, increase emissions, and slow emergency response.

## Tasks

- **single_intersection_easy** - Control 1 intersection, moderate demand, 100 steps
- **arterial_corridor_medium** - Coordinate 3 sequential intersections, 150 steps
- **urban_grid_hard** - Manage 2x2 grid under rush-hour demand, 200 steps

## Action Space

```python
phase_assignments: Dict[str, int]  # {intersection_id: phase_id}
# 0=NS_GREEN, 1=NS_YELLOW, 2=EW_GREEN, 3=EW_YELLOW, 4=ALL_RED
```

## Observation Space

Per intersection: queue lengths, wait times, current phase, emergency vehicles, pedestrian demand.
Network-wide: total waiting, avg wait time, throughput, time of day.

## Reward Function

reward = throughput_reward - wait_penalty - emergency_penalty + pedestrian_bonus
reward in [-2.0, 2.0]

## API Endpoints

- POST /reset
- POST /step
- GET /state
- GET /health
- GET /tasks

## Baseline Scores

| Task | Heuristic Score |
|------|----------------|
| single_intersection_easy | ~0.75 |
| arterial_corridor_medium | ~0.56 |
| urban_grid_hard | ~0.65 |

## Setup

```bash
docker build -t traffic-signal-env .
docker run -p 7860:7860 traffic-signal-env
```

## License

MIT