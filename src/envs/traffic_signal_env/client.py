"""
Traffic Signal Control - HTTP Environment Client

Use this in your training code to interact with the environment
running in Docker / HF Space.
"""
import json
from typing import Any, Dict, Optional
import requests


class TrafficSignalClient:
    """
    HTTP client for the Traffic Signal Control OpenEnv environment.

    Usage:
        client = TrafficSignalClient("http://localhost:8000")
        obs = client.reset(task_id="single_intersection_easy")
        result = client.step({"I0": 0})
        state = client.state()
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        """Check if server is alive."""
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(
        self,
        task_id: str = "single_intersection_easy",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Start a new episode.

        Returns:
            {
                "observation": TrafficObservation dict,
                "reward": 0.0,
                "done": False,
                "info": {...}
            }
        """
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def step(
        self,
        phase_assignments: Dict[str, int],
        task_id: Optional[str] = None,
        duration: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute one simulation tick.

        Args:
            phase_assignments: {intersection_id: phase_id}
                phase_id: 0=NS_GREEN, 1=NS_YELLOW, 2=EW_GREEN, 3=EW_YELLOW, 4=ALL_RED
            task_id: optional override
            duration: ticks to hold (1-10)

        Returns:
            {
                "observation": TrafficObservation dict,
                "reward": float,        # shaped reward this tick
                "done": bool,           # True when episode ends
                "info": {
                    "step_number": int,
                    "total_reward_so_far": float,
                    "network_throughput": int,
                    "network_avg_wait_time": float,
                    "final_score": float,   # only when done=True
                }
            }
        """
        payload: Dict[str, Any] = {
            "phase_assignments": phase_assignments,
            "duration": duration,
        }
        if task_id:
            payload["task_id"] = task_id
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        """
        Get current episode metadata.

        Returns:
            {
                "episode_id": str,
                "step_count": int,
                "task_id": str,
                "total_reward": float,
                "cumulative_wait_time": float,
                "total_throughput": int,
                "emergency_events": int,
                "num_intersections": int,
                "intersection_ids": List[str],
                "max_steps": int,
                "demand_level": str,
                "current_score": float,
            }
        """
        r = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_tasks(self) -> Dict[str, Any]:
        """List all available tasks."""
        r = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
