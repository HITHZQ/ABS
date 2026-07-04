from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from .config import ExperimentConfig
from .interfaces import CooperativeManipulationEnv, MultiAgentPolicy
from .mock_libero_coop import MockOpenDrawerPutCupSpoonEnv
from .policies import make_policy


def make_env(config: ExperimentConfig) -> CooperativeManipulationEnv:
    if config.task.env_backend == "mock":
        return MockOpenDrawerPutCupSpoonEnv(
            instruction=config.task.instruction,
            num_agents=config.task.num_agents,
            max_episode_steps=config.task.max_episode_steps,
            seed=config.task.seed,
        )
    raise NotImplementedError(
        f"env_backend={config.task.env_backend!r} is not wired yet. "
        "Implement a CooperativeManipulationEnv adapter for LIBERO/robosuite."
    )


def evaluate_episode(
    env: CooperativeManipulationEnv,
    policy: MultiAgentPolicy,
    *,
    seed: int,
) -> dict[str, Any]:
    observations = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    done = False
    final_info: dict[str, Any] = {}
    total_bandwidth = 0
    steps = 0

    while not done:
        actions = policy.act(observations)
        result = env.step(actions)
        observations = result.observations
        total_reward += result.reward
        done = result.done
        final_info = result.info
        total_bandwidth += int(getattr(policy, "bandwidth_bytes", 0))
        steps += 1

    return {
        "success": bool(final_info.get("success", False)),
        "episode_reward": total_reward,
        "steps": steps,
        "collisions": int(final_info.get("collisions", 0)),
        "bandwidth_bytes": total_bandwidth,
        "drawer_open": bool(final_info.get("drawer_open", False)),
        "cup_in_drawer": bool(final_info.get("cup_in_drawer", False)),
        "spoon_on_plate": bool(final_info.get("spoon_on_plate", False)),
    }


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot summarize zero episodes")
    return {
        "success_rate": mean(float(ep["success"]) for ep in episodes),
        "avg_reward": mean(float(ep["episode_reward"]) for ep in episodes),
        "avg_steps": mean(float(ep["steps"]) for ep in episodes),
        "avg_collisions": mean(float(ep["collisions"]) for ep in episodes),
        "avg_bandwidth_bytes": mean(float(ep["bandwidth_bytes"]) for ep in episodes),
        "drawer_open_rate": mean(float(ep["drawer_open"]) for ep in episodes),
        "cup_in_drawer_rate": mean(float(ep["cup_in_drawer"]) for ep in episodes),
        "spoon_on_plate_rate": mean(float(ep["spoon_on_plate"]) for ep in episodes),
    }


def evaluate_config(config: ExperimentConfig) -> dict[str, Any]:
    env = make_env(config)
    policy = make_policy(config)
    episodes = [
        evaluate_episode(env, policy, seed=config.task.seed + idx)
        for idx in range(config.evaluation.eval_episodes)
    ]
    summary = summarize(episodes)
    return {
        "name": config.name,
        "config": config.to_dict(),
        "summary": summary,
        "episodes": episodes,
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
