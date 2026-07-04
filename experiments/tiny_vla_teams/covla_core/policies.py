from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .interfaces import AgentAction, AgentObservation


@dataclass
class Message:
    sender: int
    mode: str
    payload: Any
    bytes_estimate: int


class CommunicationModule:
    """Communication abstraction used by the policy scaffold."""

    def __init__(self, mode: str, message_tokens: int = 8, message_dim: int = 128) -> None:
        self.mode = mode
        self.message_tokens = message_tokens
        self.message_dim = message_dim
        self.last_bandwidth_bytes = 0

    def reset(self) -> None:
        self.last_bandwidth_bytes = 0

    def encode(self, obs: AgentObservation) -> Message | None:
        if self.mode == "none":
            return None
        if self.mode == "text":
            payload = self._symbolic_text(obs)
            msg = Message(obs.agent_id, "text", payload, len(payload.encode("utf-8")))
        elif self.mode == "raw":
            # Proxy for one 224x224 RGB frame. Real experiments should report
            # actual encoded bytes or tensor bandwidth.
            msg = Message(obs.agent_id, "raw", "<rgb_frame>", 224 * 224 * 3)
        elif self.mode == "vector":
            msg = Message(obs.agent_id, "vector", obs.symbolic_state, self.message_dim * 4)
        elif self.mode == "latent":
            # Compact VLA latent-token message.
            msg = Message(
                obs.agent_id,
                "latent",
                obs.vla_latent,
                self.message_tokens * self.message_dim * 2,  # fp16 bytes
            )
        else:
            raise ValueError(f"unknown communication mode: {self.mode}")
        self.last_bandwidth_bytes += msg.bytes_estimate
        return msg

    @staticmethod
    def _symbolic_text(obs: AgentObservation) -> str:
        s = obs.symbolic_state
        return (
            f"agent={obs.agent_id}; drawer_open={s['drawer_open']}; "
            f"cup_in_drawer={s['cup_in_drawer']}; spoon_on_plate={s['spoon_on_plate']}"
        )


class ScriptedAssignment:
    """Assignment scaffold.

    In real experiments, replace this with a trainable assignment policy over
    subgoal graph nodes.  The mock implementation produces sensible subgoals
    for the representative task so all baseline arms can be smoke-tested.
    """

    def __init__(self, mode: str, num_agents: int) -> None:
        self.mode = mode
        self.num_agents = num_agents

    def assign(self, observations: list[AgentObservation]) -> dict[int, str | None]:
        state = observations[0].symbolic_state
        if self.mode == "single" or self.num_agents == 1:
            return {0: self._next_single_agent_subgoal(state)}

        if self.mode == "rule":
            return self._rule_assignment(state)

        if self.mode == "learned":
            # Placeholder for learned dynamic assignment.  This heuristic
            # intentionally reallocates cup placement after the drawer opens,
            # matching what a trained assignment policy should discover.
            return self._learned_like_assignment(state)

        raise ValueError(f"unknown assignment mode: {self.mode}")

    @staticmethod
    def _next_single_agent_subgoal(state: dict[str, Any]) -> str:
        if not state["drawer_open"]:
            return "open_drawer"
        if not state["cup_grasped"] and not state["cup_in_drawer"]:
            return "grasp_cup"
        if not state["cup_in_drawer"]:
            return "place_cup_drawer"
        if not state["spoon_grasped"] and not state["spoon_on_plate"]:
            return "grasp_spoon"
        if not state["spoon_on_plate"]:
            return "place_spoon_plate"
        return "wait"

    def _rule_assignment(self, state: dict[str, Any]) -> dict[int, str | None]:
        assignment = {i: "wait" for i in range(self.num_agents)}
        assignment[0] = self._drawer_and_cup_subgoal(state)
        if self.num_agents > 1:
            assignment[1] = self._spoon_subgoal(state)
        return assignment

    def _learned_like_assignment(self, state: dict[str, Any]) -> dict[int, str | None]:
        assignment = {i: "wait" for i in range(self.num_agents)}
        if self.num_agents == 1:
            assignment[0] = self._next_single_agent_subgoal(state)
            return assignment
        assignment[0] = self._drawer_and_cup_subgoal(state)
        assignment[1] = self._spoon_subgoal(state)
        if self.num_agents > 2 and state["drawer_open"] and not state["cup_in_drawer"]:
            assignment[2] = "place_cup_drawer" if state["cup_grasped"] else "grasp_cup"
        return assignment

    @staticmethod
    def _drawer_and_cup_subgoal(state: dict[str, Any]) -> str:
        if not state["drawer_open"]:
            return "open_drawer"
        if not state["cup_grasped"] and not state["cup_in_drawer"]:
            return "grasp_cup"
        if not state["cup_in_drawer"]:
            return "place_cup_drawer"
        return "wait"

    @staticmethod
    def _spoon_subgoal(state: dict[str, Any]) -> str:
        if not state["spoon_grasped"] and not state["spoon_on_plate"]:
            return "grasp_spoon"
        if not state["spoon_on_plate"]:
            return "place_spoon_plate"
        return "wait"


class BaselinePolicy:
    """Policy scaffold for core baselines and CoVLA ablations.

    This is not a trained VLA.  It is an executable orchestration layer that
    mirrors the experiment arms.  Real Tiny VLA / OpenVLA actors should replace
    `_subgoal_to_action` with model inference over observations, language,
    subgoals, and messages.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.name = config.name
        self.assignment = ScriptedAssignment(config.model.assignment_mode, config.task.num_agents)
        self.communication = CommunicationModule(
            config.model.communication_mode,
            message_tokens=config.model.message_tokens,
            message_dim=config.model.message_dim,
        )
        self.last_messages: list[Message] = []

    def reset(self) -> None:
        self.communication.reset()
        self.last_messages = []

    def act(self, observations: list[AgentObservation]) -> list[AgentAction]:
        self.communication.reset()
        messages = [self.communication.encode(obs) for obs in observations]
        self.last_messages = [m for m in messages if m is not None]

        assignment = self.assignment.assign(observations)
        actions: list[AgentAction] = []
        for obs in observations:
            subgoal = assignment.get(obs.agent_id) or "wait"
            actions.append(
                AgentAction(
                    agent_id=obs.agent_id,
                    symbolic_action=self._subgoal_to_action(subgoal),
                    continuous_action=None,
                )
            )
        return actions

    @staticmethod
    def _subgoal_to_action(subgoal: str) -> str:
        return subgoal if subgoal else "wait"

    @property
    def bandwidth_bytes(self) -> int:
        return self.communication.last_bandwidth_bytes


def make_policy(config: ExperimentConfig) -> BaselinePolicy:
    return BaselinePolicy(config)
