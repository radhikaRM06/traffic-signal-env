"""
Traffic Signal Control Environment - Core Implementation

Implements the OpenEnv Environment interface:
  reset() → observation dict
  step(action) → (observation, reward, done, info)
  get_state() → state dict
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Any, Dict, Optional
from .simulation import TrafficNetwork
from .graders import grade_episode


class TrafficSignalEnvironment:
    """
    Real-world traffic signal control environment.
    
    An AI agent controls traffic signal phases at urban intersections
    to minimize vehicle wait times and maximize throughput.
    
    Three tasks with increasing difficulty:
      1. single_intersection_easy  — 1 intersection, moderate demand
      2. arterial_corridor_medium  — 3 intersections in sequence, high demand
      3. urban_grid_hard           — 2×2 grid, rush-hour demand + emergencies
    """

    VALID_TASKS = [
        "single_intersection_easy",
        "arterial_corridor_medium",
        "urban_grid_hard",
    ]

    def __init__(self, task_id: str = "single_intersection_easy", seed: Optional[int] = None):
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS}")
        self.task_id = task_id
        self.seed = seed
        self.network: Optional[TrafficNetwork] = None
        self.total_reward: float = 0.0
        self._episode_stats: Dict[str, Any] = {}
        self._prev_phases: Dict[str, int] = {}
        self._green_wave_steps: int = 0
        self._pedestrian_demand_steps: int = 0
        self._pedestrian_phases_given: int = 0
        self._phase_oscillations: int = 0
        self._emergency_cleared: int = 0

    # ─── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """Start a new episode. Returns initial observation."""
        self.network = TrafficNetwork(self.task_id, self.seed)
        self.total_reward = 0.0
        self._prev_phases = {}
        self._green_wave_steps = 0
        self._pedestrian_demand_steps = 0
        self._pedestrian_phases_given = 0
        self._phase_oscillations = 0
        self._emergency_cleared = 0
        self._episode_stats = {}

        # Initial step with default phase (NS_GREEN)
        init_phases = {iid: 0 for iid in self.network.intersections}
        obs = self.network.step(init_phases)
        obs["reward"] = 0.0  # No reward on reset
        self._prev_phases = init_phases.copy()
        return obs

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one simulation tick.
        
        action dict keys:
          phase_assignments: {intersection_id: phase_id}
          duration: int (ticks — currently processed as 1 tick per call)
          task_id: str
        
        Returns observation dict with reward, done, etc.
        """
        if self.network is None:
            raise RuntimeError("Call reset() before step()")

        phase_assignments: Dict[str, int] = action.get("phase_assignments", {})
        
        # Fill missing intersections with their current phase
        for iid, intersection in self.network.intersections.items():
            if iid not in phase_assignments:
                phase_assignments[iid] = intersection.current_phase

        # Track auxiliary stats for graders
        self._track_auxiliary_stats(phase_assignments)

        obs = self.network.step(phase_assignments)
        self.total_reward += obs["reward"]

        # Update prev phases
        self._prev_phases = phase_assignments.copy()

        # If done, compute final grader score
        if obs["done"]:
            stats = self._collect_episode_stats()
            final_score = grade_episode(self.task_id, stats)
            obs["metadata"]["final_score"] = final_score
            obs["metadata"]["episode_stats"] = stats
            self._episode_stats = stats

        return obs

    def get_state(self) -> Dict[str, Any]:
        """Return current episode state/metadata."""
        if self.network is None:
            return {
                "episode_id": None,
                "step_count": 0,
                "task_id": self.task_id,
                "total_reward": 0.0,
                "cumulative_wait_time": 0.0,
                "total_throughput": 0,
                "emergency_events": 0,
                "num_intersections": 0,
                "intersection_ids": [],
                "max_steps": 100,
                "demand_level": "moderate",
                "current_score": 0.0,
            }
        state = self.network.get_state()
        state["total_reward"] = round(self.total_reward, 4)
        if self._episode_stats:
            state["current_score"] = self._episode_stats.get("final_score", 0.0)
        return state

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _track_auxiliary_stats(self, phase_assignments: Dict[str, int]):
        """Track extra metrics needed by graders."""
        if not self.network:
            return

        # Green wave: all intersections on same NS/EW axis
        phases = list(phase_assignments.values())
        if len(phases) >= 2:
            ns_phases = {0, 1}
            ew_phases = {2, 3}
            all_ns = all(p in ns_phases for p in phases)
            all_ew = all(p in ew_phases for p in phases)
            if all_ns or all_ew:
                self._green_wave_steps += 1

        # Phase oscillation: if phases flipped from last tick
        if self._prev_phases:
            for iid, phase in phase_assignments.items():
                prev = self._prev_phases.get(iid, phase)
                if abs(phase - prev) >= 2:  # big jump = oscillation
                    self._phase_oscillations += 1

        # Pedestrian demand tracking
        for iid, intersection in self.network.intersections.items():
            if intersection.pedestrian_demand:
                self._pedestrian_demand_steps += 1
                if phase_assignments.get(iid, -1) == 4:  # ALL_RED = pedestrian phase
                    self._pedestrian_phases_given += 1

        # Emergency clearance tracking (if emergency was present and now cleared)
        for iid, intersection in self.network.intersections.items():
            was_emergency = intersection.has_emergency
            # After step, if emergency cleared, count it
            # (rough proxy: if assignment gives green to emergency direction)
            if was_emergency:
                phase = phase_assignments.get(iid, -1)
                if intersection.emergency_direction in ["N", "S"] and phase == 0:
                    self._emergency_cleared += 1
                elif intersection.emergency_direction in ["E", "W"] and phase == 2:
                    self._emergency_cleared += 1

    def _collect_episode_stats(self) -> Dict[str, Any]:
        """Compile all episode stats for the grader."""
        if not self.network:
            return {}
        return {
            "total_steps": self.network.step_count,
            "cumulative_wait_time": self.network.cumulative_wait,
            "total_throughput": self.network.total_throughput,
            "emergency_events": self.network.emergency_events,
            "emergency_events_cleared": self._emergency_cleared,
            "green_wave_steps": self._green_wave_steps,
            "pedestrian_demand_steps": self._pedestrian_demand_steps,
            "pedestrian_phases_given": self._pedestrian_phases_given,
            "phase_oscillations": self._phase_oscillations,
            "total_reward": round(self.total_reward, 4),
        }
