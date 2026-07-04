from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .interfaces import AgentAction, AgentObservation, StepResult


@dataclass
class OpenDrawerState:
    drawer_open: bool = False
    cup_in_drawer: bool = False
    spoon_on_plate: bool = False
    cup_grasped: bool = False
    spoon_grasped: bool = False
    steps: int = 0
    collisions: int = 0


class MockOpenDrawerPutCupSpoonEnv:
    """Runnable mock version of the representative LIBERO-Coop task.

    This is not a robotics simulator.  It is a deterministic symbolic testbed
    that keeps the experiment code executable before LIBERO/robosuite assets
    and VLA checkpoints are wired in.

    Instruction:
        "Open the drawer, put the cup inside it, and place the spoon on the plate."

    Success:
        drawer_open and cup_in_drawer and spoon_on_plate with no blocking
        terminal failure.
    """

    valid_actions = {
        "wait",
        "open_drawer",
        "grasp_cup",
        "place_cup_drawer",
        "grasp_spoon",
        "place_spoon_plate",
    }

    def __init__(
        self,
        *,
        instruction: str,
        num_agents: int = 2,
        max_episode_steps: int = 8,
        seed: int = 0,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        self.instruction = instruction
        self.num_agents = num_agents
        self.max_episode_steps = max_episode_steps
        self._rng = random.Random(seed)
        self._state = OpenDrawerState()
        self._last_assignment: dict[int, str | None] = {}

    def reset(self, seed: int | None = None) -> list[AgentObservation]:
        if seed is not None:
            self._rng.seed(seed)
        self._state = OpenDrawerState()
        self._last_assignment = {i: None for i in range(self.num_agents)}
        return self._observations()

    def step(self, actions: list[AgentAction]) -> StepResult:
        if len(actions) != self.num_agents:
            raise ValueError(f"expected {self.num_agents} actions, got {len(actions)}")

        state = self._state
        state.steps += 1
        reward = -0.01  # time penalty
        info: dict[str, Any] = {
            "subgoal_events": [],
            "collision": False,
            "invalid_actions": [],
        }

        symbolic_actions = [a.symbolic_action for a in actions]
        for action in symbolic_actions:
            if action not in self.valid_actions:
                info["invalid_actions"].append(action)
                reward -= 0.2

        # Simple interference model: multiple non-wait actions targeting the
        # same narrow drawer workspace cause a robot-robot collision penalty.
        drawer_workspace_actions = {"open_drawer", "place_cup_drawer"}
        if sum(a in drawer_workspace_actions for a in symbolic_actions) > 1:
            state.collisions += 1
            info["collision"] = True
            reward -= 1.0

        for action in symbolic_actions:
            if action == "open_drawer" and not state.drawer_open:
                state.drawer_open = True
                reward += 1.0
                info["subgoal_events"].append("drawer_opened")
            elif action == "grasp_cup" and not state.cup_grasped and not state.cup_in_drawer:
                state.cup_grasped = True
                reward += 0.5
                info["subgoal_events"].append("cup_grasped")
            elif action == "place_cup_drawer":
                if state.drawer_open and state.cup_grasped and not state.cup_in_drawer:
                    state.cup_in_drawer = True
                    state.cup_grasped = False
                    reward += 1.0
                    info["subgoal_events"].append("cup_placed_in_drawer")
                elif not state.drawer_open:
                    reward -= 0.3
                    info["subgoal_events"].append("cup_blocked_by_closed_drawer")
            elif action == "grasp_spoon" and not state.spoon_grasped and not state.spoon_on_plate:
                state.spoon_grasped = True
                reward += 0.5
                info["subgoal_events"].append("spoon_grasped")
            elif action == "place_spoon_plate":
                if state.spoon_grasped and not state.spoon_on_plate:
                    state.spoon_on_plate = True
                    state.spoon_grasped = False
                    reward += 1.0
                    info["subgoal_events"].append("spoon_placed_on_plate")

        success = state.drawer_open and state.cup_in_drawer and state.spoon_on_plate
        done = success or state.steps >= self.max_episode_steps
        if success:
            reward += 10.0

        info.update(
            {
                "success": success,
                "drawer_open": state.drawer_open,
                "cup_in_drawer": state.cup_in_drawer,
                "spoon_on_plate": state.spoon_on_plate,
                "steps": state.steps,
                "collisions": state.collisions,
            }
        )
        return StepResult(self._observations(), reward, done, info)

    def _observations(self) -> list[AgentObservation]:
        state_dict = {
            "drawer_open": self._state.drawer_open,
            "cup_in_drawer": self._state.cup_in_drawer,
            "spoon_on_plate": self._state.spoon_on_plate,
            "cup_grasped": self._state.cup_grasped,
            "spoon_grasped": self._state.spoon_grasped,
            "steps": self._state.steps,
            "collisions": self._state.collisions,
        }
        return [
            AgentObservation(
                agent_id=i,
                instruction=self.instruction,
                subgoal=self._last_assignment.get(i),
                symbolic_state=dict(state_dict),
                # Stand-in for a VLA latent. Real environments should supply
                # tensors from the VLA encoder/adapter.
                vla_latent=tuple(sorted(state_dict.items())),
            )
            for i in range(self.num_agents)
        ]

    def set_assignment(self, assignment: dict[int, str | None]) -> None:
        self._last_assignment.update(assignment)
