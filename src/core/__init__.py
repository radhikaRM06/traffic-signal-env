"""
OpenEnv Core: Base classes for Environment and HTTP Client.
Follows the OpenEnv spec from meta-pytorch/OpenEnv.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel


# ─── Base Models (Pydantic for FastAPI compatibility) ─────────────────────────

class Action(BaseModel):
    """Base class for all actions."""
    metadata: Dict[str, Any] = {}


class Observation(BaseModel):
    """Base class for all observations."""
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = {}


class State(BaseModel):
    """Base class for episode state/metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0


# ─── StepResult ───────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Returned by step() and reset()."""
    observation: Dict[str, Any]
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


# ─── Abstract Environment ─────────────────────────────────────────────────────

class Environment(ABC):
    """Base class for all OpenEnv environment implementations."""

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Start a new episode. Returns initial observation dict."""
        ...

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action. Returns (observation, reward, done, info) as dict."""
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return current episode state/metadata."""
        ...
