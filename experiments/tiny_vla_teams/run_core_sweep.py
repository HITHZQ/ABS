#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from covla_core.config import baseline_sweep
from covla_core.evaluate import evaluate_config, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run core Tiny VLA Teams baseline/ablation sweep.")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of robot agents in the team.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per experiment arm.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/core_sweep.jsonl"),
        help="JSONL output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for config in baseline_sweep(num_agents=args.num_agents, seed=args.seed):
        config = config.__class__(
            name=config.name,
            task=config.task,
            model=config.model,
            training=config.training,
            evaluation=config.evaluation.__class__(eval_episodes=args.eval_episodes),
        )
        result = evaluate_config(config)
        rows.append(result)
        print(json.dumps({"name": result["name"], **result["summary"]}, sort_keys=True))
    write_jsonl(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
