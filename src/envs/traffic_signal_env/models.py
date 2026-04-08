"""
Traffic Signal Control Environment - Type-Safe Models
Uses pydantic when available (server), falls back to dataclasses.
"""
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    class TrafficAction(BaseModel):
        phase_assignments: Dict[str, int] = Field(default_factory=dict)
        duration: int = Field(default=5, ge=1, le=10)
        task_id: str = "single_intersection_easy"
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class IntersectionObservation(BaseModel):
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

    class TrafficObservation(BaseModel):
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)
        intersections: List[IntersectionObservation] = Field(default_factory=list)
        total_vehicles_waiting: int = 0
        total_vehicles_in_network: int = 0
        network_avg_wait_time: float = 0.0
        network_throughput: int = 0
        sim_time: int = 0
        time_of_day: str = "morning"
        task_id: str = "single_intersection_easy"
        step_number: int = 0
        legal_phases: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

    class TrafficState(BaseModel):
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

except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class TrafficAction:
        phase_assignments: Dict[str, int] = field(default_factory=dict)
        duration: int = 5
        task_id: str = "single_intersection_easy"
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class IntersectionObservation:
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

    @dataclass
    class TrafficObservation:
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)
        intersections: List[Any] = field(default_factory=list)
        total_vehicles_waiting: int = 0
        total_vehicles_in_network: int = 0
        network_avg_wait_time: float = 0.0
        network_throughput: int = 0
        sim_time: int = 0
        time_of_day: str = "morning"
        task_id: str = "single_intersection_easy"
        step_number: int = 0
        legal_phases: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    @dataclass
    class TrafficState:
        episode_id: Optional[str] = None
        step_count: int = 0
        task_id: str = "single_intersection_easy"
        total_reward: float = 0.0
        cumulative_wait_time: float = 0.0
        total_throughput: int = 0
        emergency_events: int = 0
        num_intersections: int = 1
        intersection_ids: List[str] = field(default_factory=list)
        max_steps: int = 100
        demand_level: str = "moderate"
        current_score: float = 0.0
