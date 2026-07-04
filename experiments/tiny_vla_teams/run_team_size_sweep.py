#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from covla_core.config import EvaluationConfig, ExperimentConfig, ModelConfig, TaskConfig, TrainingConfig
from covla_core.evaluate import evaluate_config, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run team-size scaling scaffold for CoVLA.")
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--max-agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/team_size_sweep.jsonl"),
        help="JSONL output path.",
    )
    return parser.parse_args()


def config_for_team_size(num_agents: int, seed: int, eval_episodes: int) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"covla_full_n{num_agents}",
        task=TaskConfig(num_agents=num_agents, seed=seed),
        model=ModelConfig(
            policy_kind="covla",
            communication_mode="latent",
            assignment_mode="learned",
            train_critic=True,
            train_assignment=True,
            train_communication=True,
        ),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(eval_episodes=eval_episodes),
    )


def main() -> None:
    args = parse_args()
    if args.min_agents < 1 or args.max_agents < args.min_agents:
        raise SystemExit("invalid agent range")

    rows = []
    for n in range(args.min_agents, args.max_agents + 1):
        result = evaluate_config(config_for_team_size(n, args.seed, args.eval_episodes))
        rows.append(result)
        print(json.dumps({"team_size": n, **result["summary"]}, sort_keys=True))
    write_jsonl(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
