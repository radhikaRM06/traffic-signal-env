"""
Traffic Signal Control — Environment Implementation
Extends openenv-core Environment base class.
"""
import uuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Optional

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:
        pass

from models import TrafficAction, TrafficObservation, TrafficState, IntersectionObs
from server.simulation import TrafficNetwork
from server.graders import grade_episode


class TrafficSignalEnvironment(Environment):
    """
    Real-world traffic signal control environment.
    Controls urban intersection signal phases to minimize wait times.

    Tasks:
      single_intersection_easy  — 1 intersection, moderate demand
      arterial_corridor_medium  — 3 intersections, high demand
      urban_grid_hard           — 2x2 grid, rush-hour + emergencies
    """

    VALID_TASKS = [
        "single_intersection_easy",
        "arterial_corridor_medium",
        "urban_grid_hard",
    ]

    def __init__(self):
        self._task_id = "single_intersection_easy"
        self._seed = 42
        self._network: Optional[TrafficNetwork] = None
        self._total_reward: float = 0.0
        self._prev_phases = {}
        self._green_wave_steps = 0
        self._pedestrian_demand_steps = 0
        self._pedestrian_phases_given = 0
        self._phase_oscillations = 0
        self._emergency_cleared = 0
        self._state = TrafficState()

    def reset(self, task_id: str = None, seed: int = None) -> TrafficObservation:
        """Start a new episode."""
        if task_id and task_id in self.VALID_TASKS:
            self._task_id = task_id
        if seed is not None:
            self._seed = seed

        self._network = TrafficNetwork(self._task_id, self._seed)
        self._total_reward = 0.0
        self._prev_phases = {}
        self._green_wave_steps = 0
        self._pedestrian_demand_steps = 0
        self._pedestrian_phases_given = 0
        self._phase_oscillations = 0
        self._emergency_cleared = 0

        # Update state
        self._state = TrafficState(
            episode_id=str(uuid.uuid4())[:8],
            step_count=0,
            task_id=self._task_id,
            num_intersections=len(self._network.intersections),
            intersection_ids=list(self._network.intersections.keys()),
            max_steps=self._network.max_steps,
            demand_level=self._network.demand_level,
        )

        # Initial tick
        init_phases = {iid: 0 for iid in self._network.intersections}
        raw = self._network.step(init_phases)
        raw["reward"] = 0.0
        self._prev_phases = init_phases.copy()
        return self._build_observation(raw)

    def step(self, action: TrafficAction) -> TrafficObservation:
        """Execute one simulation tick."""
        if self._network is None:
            raise RuntimeError("Call reset() first")

        phase_assignments = dict(action.phase_assignments) if action.phase_assignments else {}

        # Fill missing intersections
        for iid, intersection in self._network.intersections.items():
            if iid not in phase_assignments:
                phase_assignments[iid] = intersection.current_phase

        self._track_aux(phase_assignments)
        raw = self._network.step(phase_assignments)
        self._total_reward += raw["reward"]
        self._prev_phases = phase_assignments.copy()

        # Update state
        self._state.step_count = self._network.step_count
        self._state.total_reward = round(self._total_reward, 4)
        self._state.cumulative_wait_time = round(self._network.cumulative_wait, 2)
        self._state.total_throughput = self._network.total_throughput
        self._state.emergency_events = self._network.emergency_events

        if raw["done"]:
            stats = self._collect_stats()
            score = grade_episode(self._task_id, stats)
            raw["metadata"]["final_score"] = score
            raw["metadata"]["episode_stats"] = stats
            self._state.current_score = score

        return self._build_observation(raw)

    @property
    def state(self) -> TrafficState:
        return self._state

    def _build_observation(self, raw: dict) -> TrafficObservation:
        intersections = [IntersectionObs(**i) for i in raw.get("intersections", [])]
        return TrafficObservation(
            done=raw.get("done", False),
            reward=raw.get("reward", 0.0),
            intersections=intersections,
            total_vehicles_waiting=raw.get("total_vehicles_waiting", 0),
            network_avg_wait_time=raw.get("network_avg_wait_time", 0.0),
            network_throughput=raw.get("network_throughput", 0),
            sim_time=raw.get("sim_time", 0),
            time_of_day=raw.get("time_of_day", "morning"),
            task_id=raw.get("task_id", self._task_id),
            step_number=raw.get("step_number", 0),
            legal_phases=raw.get("legal_phases", [0, 1, 2, 3, 4]),
            metadata=raw.get("metadata", {}),
        )

    def _track_aux(self, phase_assignments):
        if not self._network:
            return
        phases = list(phase_assignments.values())
        if len(phases) >= 2:
            ns = {0, 1}
            ew = {2, 3}
            if all(p in ns for p in phases) or all(p in ew for p in phases):
                self._green_wave_steps += 1
        if self._prev_phases:
            for iid, phase in phase_assignments.items():
                prev = self._prev_phases.get(iid, phase)
                if abs(phase - prev) >= 2:
                    self._phase_oscillations += 1
        for iid, intersection in self._network.intersections.items():
            if intersection.pedestrian_demand:
                self._pedestrian_demand_steps += 1
                if phase_assignments.get(iid, -1) == 4:
                    self._pedestrian_phases_given += 1
            if intersection.has_emergency:
                phase = phase_assignments.get(iid, -1)
                if intersection.emergency_direction in ["N", "S"] and phase == 0:
                    self._emergency_cleared += 1
                elif intersection.emergency_direction in ["E", "W"] and phase == 2:
                    self._emergency_cleared += 1

    def _collect_stats(self):
        if not self._network:
            return {}
        return {
            "total_steps": self._network.step_count,
            "cumulative_wait_time": self._network.cumulative_wait,
            "total_throughput": self._network.total_throughput,
            "emergency_events": self._network.emergency_events,
            "emergency_events_cleared": self._emergency_cleared,
            "green_wave_steps": self._green_wave_steps,
            "pedestrian_demand_steps": self._pedestrian_demand_steps,
            "pedestrian_phases_given": self._pedestrian_phases_given,
            "phase_oscillations": self._phase_oscillations,
            "total_reward": round(self._total_reward, 4),
        }
