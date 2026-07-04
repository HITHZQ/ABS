from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class AgentObservation:
    """Observation passed to one agent.

    Real backends should fill `rgb`, `proprio`, and `vla_latent`.  The mock
    backend uses symbolic fields so the experiment orchestration is runnable
    without robotics dependencies.
    """

    agent_id: int
    instruction: str
    subgoal: str | None
    symbolic_state: dict[str, Any]
    rgb: Any | None = None
    proprio: Any | None = None
    vla_latent: Any | None = None


@dataclass
class AgentAction:
    """One agent action.

    `continuous_action` is the real robot action slot, e.g. 7-DoF delta pose
    plus gripper.  `symbolic_action` is used only by the mock environment and
    by high-level subgoal debugging.
    """

    agent_id: int
    symbolic_action: str
    continuous_action: Any | None = None


@dataclass
class StepResult:
    observations: list[AgentObservation]
    reward: float
    done: bool
    info: dict[str, Any]


class CooperativeManipulationEnv(Protocol):
    """Minimal interface expected from LIBERO-Coop-like environments."""

    num_agents: int

    def reset(self, seed: int | None = None) -> list[AgentObservation]:
        ...

    def step(self, actions: list[AgentAction]) -> StepResult:
        ...


class MultiAgentPolicy(Protocol):
    """Policy interface for both baselines and CoVLA variants."""

    name: str

    def reset(self) -> None:
        ...

    def act(self, observations: list[AgentObservation]) -> list[AgentAction]:
        ...


class MetricSink(Protocol):
    """Receives per-episode metric dictionaries."""

    def write(self, metrics: dict[str, Any]) -> None:
        ...
