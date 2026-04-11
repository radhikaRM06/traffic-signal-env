"""
Traffic Signal Control - Task Graders

Each grader produces a score in [0.0, 1.0] evaluating agent performance
on a completed episode. Graders are deterministic given episode stats.
"""
from typing import Dict, Any


def _normalize(value: float, worst: float, best: float) -> float:
    """Linearly normalize value between [0, 1]. worst=0, best=1."""
    if best == worst:
        return 0.5
    normalized = (value - worst) / (best - worst)
    return max(0.0, min(1.0, normalized))


# ─── Task 1: Single Intersection (Easy) ───────────────────────────────────────

def grade_single_intersection(episode_stats: Dict[str, Any]) -> float:
    """
    Score an episode on the single-intersection task.
    
    Criteria:
    - Primary (60%): Average wait time reduction vs do-nothing baseline
    - Secondary (25%): Total throughput
    - Bonus (15%): Emergency vehicle clearance rate
    
    Returns score in [0.0, 1.0]
    """
    steps = episode_stats.get("total_steps", 100)
    cumulative_wait = episode_stats.get("cumulative_wait_time", 0.0)
    total_throughput = episode_stats.get("total_throughput", 0)
    emergency_events = episode_stats.get("emergency_events", 0)
    emergency_cleared = episode_stats.get("emergency_events_cleared", 0)

    # Do-nothing baseline: avg_wait ≈ 15 per step (vehicle-ticks)
    # Good agent: avg_wait ≈ 3 per step
    avg_wait_per_step = cumulative_wait / max(1, steps)
    wait_score = _normalize(avg_wait_per_step, worst=20.0, best=1.0)

    # Throughput: baseline ~8/step, good ~18/step
    avg_throughput = total_throughput / max(1, steps)
    throughput_score = _normalize(avg_throughput, worst=2.0, best=20.0)

    # Emergency clearance
    if emergency_events == 0:
        emergency_score = 1.0
    else:
        emergency_score = emergency_cleared / emergency_events

    score = (
        0.60 * wait_score +
        0.25 * throughput_score +
        0.15 * emergency_score
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─── Task 2: Arterial Corridor (Medium) ───────────────────────────────────────

def grade_arterial_corridor(episode_stats: Dict[str, Any]) -> float:
    """
    Score an episode on the 3-intersection arterial corridor task.
    
    Criteria:
    - Primary (50%): Network average wait time
    - Secondary (30%): Network throughput (green wave efficiency)
    - Coordination (20%): Phase coordination score (green wave bonus)
    
    Returns score in [0.0, 1.0]
    """
    steps = episode_stats.get("total_steps", 150)
    cumulative_wait = episode_stats.get("cumulative_wait_time", 0.0)
    total_throughput = episode_stats.get("total_throughput", 0)
    green_wave_steps = episode_stats.get("green_wave_steps", 0)

    avg_wait_per_step = cumulative_wait / max(1, steps)
    wait_score = _normalize(avg_wait_per_step, worst=30.0, best=2.0)

    avg_throughput = total_throughput / max(1, steps)
    throughput_score = _normalize(avg_throughput, worst=3.0, best=45.0)

    # Green wave: fraction of steps where all 3 intersections had same NS/EW phase
    coordination_score = green_wave_steps / max(1, steps)

    score = (
        0.50 * wait_score +
        0.30 * throughput_score +
        0.20 * coordination_score
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─── Task 3: Urban Grid (Hard) ────────────────────────────────────────────────

def grade_urban_grid(episode_stats: Dict[str, Any]) -> float:
    """
    Score an episode on the 2x2 urban grid task.
    
    Criteria:
    - Primary (40%): Network average wait time under rush-hour demand
    - Secondary (25%): Total throughput
    - Emergency (20%): Emergency vehicle clearance speed
    - Pedestrian (10%): Pedestrian phase utilization
    - Stability (5%): Penalty for oscillating phases (flipping every tick)
    
    Returns score in [0.0, 1.0]
    """
    steps = episode_stats.get("total_steps", 200)
    cumulative_wait = episode_stats.get("cumulative_wait_time", 0.0)
    total_throughput = episode_stats.get("total_throughput", 0)
    emergency_events = episode_stats.get("emergency_events", 0)
    emergency_cleared = episode_stats.get("emergency_events_cleared", 0)
    pedestrian_phases_given = episode_stats.get("pedestrian_phases_given", 0)
    pedestrian_demand_steps = episode_stats.get("pedestrian_demand_steps", 1)
    phase_oscillations = episode_stats.get("phase_oscillations", 0)

    # Wait time (harder task, higher baseline)
    avg_wait_per_step = cumulative_wait / max(1, steps)
    wait_score = _normalize(avg_wait_per_step, worst=60.0, best=5.0)

    # Throughput
    avg_throughput = total_throughput / max(1, steps)
    throughput_score = _normalize(avg_throughput, worst=4.0, best=60.0)

    # Emergency
    if emergency_events == 0:
        emergency_score = 1.0
    else:
        emergency_score = min(1.0, emergency_cleared / emergency_events)

    # Pedestrian utilization
    if pedestrian_demand_steps == 0:
        pedestrian_score = 1.0
    else:
        pedestrian_score = min(1.0, pedestrian_phases_given / pedestrian_demand_steps)

    # Phase stability (penalize rapid oscillation)
    oscillation_rate = phase_oscillations / max(1, steps)
    stability_score = max(0.0, 1.0 - oscillation_rate * 2)

    score = (
        0.40 * wait_score +
        0.25 * throughput_score +
        0.20 * emergency_score +
        0.10 * pedestrian_score +
        0.05 * stability_score
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─── Router ───────────────────────────────────────────────────────────────────

GRADERS = {
    "single_intersection_easy": grade_single_intersection,
    "arterial_corridor_medium": grade_arterial_corridor,
    "urban_grid_hard": grade_urban_grid,
}


def grade_episode(task_id: str, episode_stats: Dict[str, Any]) -> float:
    """Route to the correct grader and return score [0.0, 1.0]."""
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(GRADERS.keys())}")
    return grader(episode_stats)
