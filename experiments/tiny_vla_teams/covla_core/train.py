from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


def train_config(config: ExperimentConfig, output_dir: str | Path) -> dict[str, Any]:
    """Training entry point scaffold.

    The real implementation should perform:
    1. behavior cloning initialization of subgoal-conditioned VLA actors;
    2. critic pretraining on demonstration and rollout trajectories;
    3. MARL fine-tuning of adapters, communication, assignment, and critic;
    4. checkpoint export for evaluation.

    This scaffold writes a manifest so experiment orchestration can be tested
    before GPU-heavy VLA/MARL code is integrated.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": config.name,
        "status": "scaffold_only",
        "message": (
            "Replace covla_core.train.train_config with BC + MAPPO fine-tuning "
            "when LIBERO-Coop and VLA checkpoints are available."
        ),
        "config": config.to_dict(),
        "recommended_order": [
            "bc_subgoal_vla",
            "critic_pretrain",
            "communication_auxiliary_losses",
            "mappo_adapter_finetune",
            "frozen_backbone_evaluation",
        ],
    }
    path = out / f"{config.name}_train_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest
