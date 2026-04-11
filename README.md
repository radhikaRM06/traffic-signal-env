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

## Tasks

- **single_intersection_easy** - 1 intersection, moderate demand, 100 steps
- **arterial_corridor_medium** - 3 sequential intersections, high demand, 150 steps
- **urban_grid_hard** - 2x2 grid, rush-hour demand + emergencies, 200 steps

## API

- `POST /reset` - Start new episode
- `POST /step` - Execute action
- `GET /state` - Episode metadata
- `GET /health` - Health check
- `GET /tasks` - List tasks

## Action Space

```json
{"phase_assignments": {"I0": 0}, "duration": 5, "task_id": "single_intersection_easy"}
```

Phases: 0=NS_GREEN, 1=NS_YELLOW, 2=EW_GREEN, 3=EW_YELLOW, 4=ALL_RED

## Reward

```
reward = throughput_bonus - wait_penalty - emergency_penalty + pedestrian_bonus
range: [-2.0, 2.0]
```

## Baseline Scores

| Task | Score |
|------|-------|
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
