"""
Traffic Signal Control — Models
Uses openenv-core Pydantic types when available, falls back to plain pydantic.
"""
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from pydantic import BaseModel as Action
    from pydantic import BaseModel as Observation
    from pydantic import BaseModel as State

from pydantic import BaseModel, Field


class TrafficAction(Action):
    """
    Action to set signal phases for controlled intersections.
    phase_assignments: {intersection_id: phase_id}
      0 = NS_GREEN, 1 = NS_YELLOW, 2 = EW_GREEN, 3 = EW_YELLOW, 4 = ALL_RED
    """
    phase_assignments: Dict[str, int] = Field(
        default_factory=dict,
        description="Map of intersection_id -> phase_id (0-4)"
    )
    duration: int = Field(default=5, ge=1, le=10)
    task_id: str = Field(default="single_intersection_easy")


class IntersectionObs(BaseModel):
    intersection_id: str = ""
    current_phase: int = 0
    phase_time_elapsed: int = 0
    queue_north: int = 0
    queue_south: int = 0
    queue_east: int = 0
    queue_west: int = 0
    avg_wait_north: float = 0.0
    avg_wait_south: float = 0.0
    avg_wait_east: float = 0.0
    avg_wait_west: float = 0.0
    throughput_last_tick: int = 0
    emergency_vehicle_present: bool = False
    emergency_vehicle_direction: Optional[str] = None
    pedestrian_demand: bool = False


class TrafficObservation(Observation):
    """Full observation returned after each step."""
    done: bool = False
    reward: float = 0.0
    intersections: List[IntersectionObs] = Field(default_factory=list)
    total_vehicles_waiting: int = 0
    network_avg_wait_time: float = 0.0
    network_throughput: int = 0
    sim_time: int = 0
    time_of_day: str = "morning"
    task_id: str = "single_intersection_easy"
    step_number: int = 0
    legal_phases: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrafficState(State):
    """Episode metadata returned by GET /state."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = "single_intersection_easy"
    total_reward: float = 0.0
    cumulative_wait_time: float = 0.0
    total_throughput: int = 0
    emergency_events: int = 0
    num_intersections: int = 1
    intersection_ids: List[str] = Field(default_factory=list)
    max_steps: int = 100
    demand_level: str = "moderate"
    current_score: float = 0.0
