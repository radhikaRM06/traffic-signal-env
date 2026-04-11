"""
Traffic Signal Simulation Engine

Simulates realistic vehicle queuing, arrivals, and throughput at
urban intersections using a discrete-event model.
"""
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Phase Definitions ────────────────────────────────────────────────────────

PHASE_DEFINITIONS = {
    0: {"name": "NS_GREEN",   "ns_green": True,  "ew_green": False, "all_red": False},
    1: {"name": "NS_YELLOW",  "ns_green": False, "ew_green": False, "all_red": False},
    2: {"name": "EW_GREEN",   "ns_green": False, "ew_green": True,  "all_red": False},
    3: {"name": "EW_YELLOW",  "ns_green": False, "ew_green": False, "all_red": False},
    4: {"name": "ALL_RED",    "ns_green": False, "ew_green": False, "all_red": True},
}

# Saturation flow rate: max vehicles that can pass per tick per approach
SATURATION_FLOW = 3

# Minimum green time in ticks (safety constraint)
MIN_GREEN_TICKS = 2


@dataclass
class Approach:
    """One directional approach to an intersection."""
    direction: str  # N, S, E, W
    queue: int = 0
    total_wait: float = 0.0
    vehicles_served: int = 0
    arrival_rate: float = 0.5  # vehicles per tick (Poisson mean)

    def arrive(self, demand_multiplier: float = 1.0) -> int:
        """Simulate vehicle arrivals (Poisson process)."""
        mean = self.arrival_rate * demand_multiplier
        arrivals = min(int(random.expovariate(1.0 / mean)) if mean > 0 else 0, 8)
        self.queue += arrivals
        return arrivals

    def serve(self, green: bool) -> int:
        """Serve vehicles if green. Returns vehicles that passed."""
        if not green or self.queue == 0:
            # Vehicles are waiting — accrue wait time
            self.total_wait += self.queue
            return 0
        served = min(self.queue, SATURATION_FLOW)
        self.queue -= served
        self.vehicles_served += served
        # Remaining vehicles still wait
        self.total_wait += self.queue
        return served

    @property
    def avg_wait(self) -> float:
        if self.vehicles_served == 0:
            return self.total_wait / max(1, self.queue + 1)
        return self.total_wait / max(1, self.vehicles_served)


@dataclass
class Intersection:
    """Simulates a single 4-way signalized intersection."""
    intersection_id: str
    approaches: Dict[str, Approach] = field(default_factory=dict)
    current_phase: int = 0
    phase_time_elapsed: int = 0
    has_emergency: bool = False
    emergency_direction: Optional[str] = None
    pedestrian_demand: bool = False
    throughput_last_tick: int = 0

    def __post_init__(self):
        if not self.approaches:
            self.approaches = {
                "N": Approach("N", arrival_rate=0.6),
                "S": Approach("S", arrival_rate=0.6),
                "E": Approach("E", arrival_rate=0.5),
                "W": Approach("W", arrival_rate=0.5),
            }

    def set_arrival_rates(self, n=0.6, s=0.6, e=0.5, w=0.5):
        self.approaches["N"].arrival_rate = n
        self.approaches["S"].arrival_rate = s
        self.approaches["E"].arrival_rate = e
        self.approaches["W"].arrival_rate = w

    def step(self, phase: int, demand_multiplier: float = 1.0) -> Tuple[int, float]:
        """
        Advance one simulation tick.
        Returns (throughput, avg_wait_this_tick)
        """
        # Update phase
        if phase != self.current_phase:
            self.current_phase = phase
            self.phase_time_elapsed = 0
        else:
            self.phase_time_elapsed += 1

        phase_def = PHASE_DEFINITIONS[phase]
        ns_green = phase_def["ns_green"]
        ew_green = phase_def["ew_green"]

        # Random emergency vehicle spawn (1% chance)
        if random.random() < 0.01 and not self.has_emergency:
            self.has_emergency = True
            self.emergency_direction = random.choice(["N", "S", "E", "W"])

        # Random pedestrian demand (10% chance)
        self.pedestrian_demand = random.random() < 0.10

        # Arrivals
        for direction, approach in self.approaches.items():
            approach.arrive(demand_multiplier)

        # Serving (green = can pass)
        served_ns = self.approaches["N"].serve(ns_green) + self.approaches["S"].serve(ns_green)
        served_ew = self.approaches["E"].serve(ew_green) + self.approaches["W"].serve(ew_green)
        self.throughput_last_tick = served_ns + served_ew

        # Emergency vehicle clears when its direction gets green
        if self.has_emergency:
            dir_to_axis = {"N": "ns", "S": "ns", "E": "ew", "W": "ew"}
            axis = dir_to_axis.get(self.emergency_direction, "ns")
            if (axis == "ns" and ns_green) or (axis == "ew" and ew_green):
                self.has_emergency = False
                self.emergency_direction = None

        total_queue = sum(a.queue for a in self.approaches.values())
        avg_wait = sum(a.avg_wait for a in self.approaches.values()) / 4

        return self.throughput_last_tick, avg_wait

    def to_obs_dict(self) -> dict:
        n, s, e, w = (self.approaches[d] for d in ["N", "S", "E", "W"])
        return {
            "intersection_id": self.intersection_id,
            "current_phase": self.current_phase,
            "phase_time_elapsed": self.phase_time_elapsed,
            "queue_north": n.queue,
            "queue_south": s.queue,
            "queue_east": e.queue,
            "queue_west": w.queue,
            "avg_wait_north": round(n.avg_wait, 2),
            "avg_wait_south": round(s.avg_wait, 2),
            "avg_wait_east": round(e.avg_wait, 2),
            "avg_wait_west": round(w.avg_wait, 2),
            "throughput_last_tick": self.throughput_last_tick,
            "emergency_vehicle_present": self.has_emergency,
            "emergency_vehicle_direction": self.emergency_direction,
            "pedestrian_demand": self.pedestrian_demand,
        }


# ─── Network ──────────────────────────────────────────────────────────────────

class TrafficNetwork:
    """
    Manages a network of intersections for a given task.
    """

    def __init__(self, task_id: str, seed: Optional[int] = None):
        self.task_id = task_id
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.intersections: Dict[str, Intersection] = {}
        self.sim_time: int = 0
        self.step_count: int = 0
        self.episode_id: str = str(uuid.uuid4())[:8]
        self.total_throughput: int = 0
        self.cumulative_wait: float = 0.0
        self.emergency_events: int = 0
        self.max_steps: int = 100
        self.demand_level: str = "moderate"

        self._setup_task(task_id)

    def _setup_task(self, task_id: str):
        """Configure intersections based on task."""
        if task_id == "single_intersection_easy":
            self.max_steps = 100
            self.demand_level = "moderate"
            i = Intersection("I0")
            i.set_arrival_rates(n=0.5, s=0.5, e=0.4, w=0.4)
            self.intersections["I0"] = i

        elif task_id == "arterial_corridor_medium":
            self.max_steps = 150
            self.demand_level = "high"
            # 3 intersections in a line: I0 → I1 → I2
            rates = [(0.7, 0.7, 0.4, 0.4), (0.6, 0.6, 0.5, 0.5), (0.5, 0.5, 0.6, 0.6)]
            for idx, (n, s, e, w) in enumerate(rates):
                i = Intersection(f"I{idx}")
                i.set_arrival_rates(n, s, e, w)
                self.intersections[f"I{idx}"] = i

        elif task_id == "urban_grid_hard":
            self.max_steps = 200
            self.demand_level = "rush_hour"
            # 2x2 grid: I00, I01, I10, I11
            grid_ids = ["I00", "I01", "I10", "I11"]
            rush_rates = [
                (0.9, 0.9, 0.6, 0.6),
                (0.8, 0.8, 0.7, 0.7),
                (0.7, 0.7, 0.8, 0.8),
                (0.6, 0.6, 0.9, 0.9),
            ]
            for iid, (n, s, e, w) in zip(grid_ids, rush_rates):
                i = Intersection(iid)
                i.set_arrival_rates(n, s, e, w)
                self.intersections[iid] = i
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    def _demand_multiplier(self) -> float:
        """Return demand multiplier based on time of day."""
        level_map = {"low": 0.5, "moderate": 1.0, "high": 1.4, "rush_hour": 1.8}
        return level_map.get(self.demand_level, 1.0)

    def time_of_day(self) -> str:
        progress = self.step_count / self.max_steps
        if progress < 0.25:
            return "morning"
        elif progress < 0.5:
            return "afternoon"
        elif progress < 0.75:
            return "evening"
        return "night"

    def step(self, phase_assignments: Dict[str, int]) -> dict:
        """
        Advance the simulation one tick.
        phase_assignments: {intersection_id: phase_id}
        Returns raw observation dict.
        """
        self.sim_time += 1
        self.step_count += 1
        multiplier = self._demand_multiplier()

        tick_throughput = 0
        tick_wait = 0.0
        intersection_obs = []

        for iid, intersection in self.intersections.items():
            phase = phase_assignments.get(iid, intersection.current_phase)
            # Clamp phase to valid range
            phase = max(0, min(4, int(phase)))
            tp, aw = intersection.step(phase, multiplier)
            tick_throughput += tp
            tick_wait += aw

            if intersection.has_emergency:
                self.emergency_events += 1

            intersection_obs.append(intersection.to_obs_dict())

        self.total_throughput += tick_throughput
        self.cumulative_wait += tick_wait

        n_intersections = len(self.intersections)
        avg_wait = tick_wait / n_intersections if n_intersections else 0.0

        total_waiting = sum(
            sum(a.queue for a in i.approaches.values())
            for i in self.intersections.values()
        )

        done = self.step_count >= self.max_steps

        # ── Reward Shaping ──────────────────────────────────────────────────
        # +throughput bonus
        throughput_reward = tick_throughput * 0.3
        # -wait penalty (normalized by number of intersections & max possible queue)
        max_possible_wait = n_intersections * 4 * 20.0  # 20 vehicles/approach max
        wait_penalty = (tick_wait / max(1.0, max_possible_wait)) * 2.0
        # -emergency penalty (if emergency not cleared quickly)
        emergency_penalty = 0.5 if any(
            i.has_emergency for i in self.intersections.values()
        ) else 0.0
        # +pedestrian bonus (if ALL_RED phase given when pedestrian demand)
        ped_bonus = 0.1 if any(
            i.pedestrian_demand and phase_assignments.get(i.intersection_id, 0) == 4
            for i in self.intersections.values()
        ) else 0.0

        reward = throughput_reward - wait_penalty - emergency_penalty + ped_bonus
        reward = round(max(-2.0, min(2.0, reward)), 4)  # clip

        return {
            "done": done,
            "reward": reward,
            "intersections": intersection_obs,
            "total_vehicles_waiting": total_waiting,
            "total_vehicles_in_network": total_waiting + tick_throughput,
            "network_avg_wait_time": round(avg_wait, 2),
            "network_throughput": tick_throughput,
            "sim_time": self.sim_time,
            "time_of_day": self.time_of_day(),
            "task_id": self.task_id,
            "step_number": self.step_count,
            "legal_phases": [0, 1, 2, 3, 4],
            "metadata": {
                "demand_level": self.demand_level,
                "episode_id": self.episode_id,
            },
        }

    def reset(self) -> dict:
        """Reset environment to initial state."""
        self.__init__(self.task_id, self.seed)
        # Return initial obs
        phase_assignments = {iid: 0 for iid in self.intersections}
        return self.step(phase_assignments)

    def get_state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "task_id": self.task_id,
            "total_reward": 0.0,  # tracked externally
            "cumulative_wait_time": round(self.cumulative_wait, 2),
            "total_throughput": self.total_throughput,
            "emergency_events": self.emergency_events,
            "num_intersections": len(self.intersections),
            "intersection_ids": list(self.intersections.keys()),
            "max_steps": self.max_steps,
            "demand_level": self.demand_level,
            "current_score": 0.0,
        }
