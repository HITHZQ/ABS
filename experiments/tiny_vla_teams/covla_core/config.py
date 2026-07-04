from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


CommunicationMode = Literal["none", "text", "raw", "vector", "latent"]
AssignmentMode = Literal["single", "rule", "learned"]
PolicyKind = Literal["tiny_vla", "large_vla", "large_vla_centralized", "mappo_visual", "covla"]


@dataclass(frozen=True)
class TaskConfig:
    """Benchmark task configuration.

    The default task is the representative OpenDrawer-PutCupAndSpoon example.
    Replace `env_backend="mock"` with "libero" or "robosuite" when wiring in a
    real manipulation simulator.
    """

    env_backend: str = "mock"
    task_family: str = "open_container_place"
    task_name: str = "OpenDrawer-PutCupAndSpoon"
    instruction: str = "Open the drawer, put the cup inside it, and place the spoon on the plate."
    num_agents: int = 2
    max_episode_steps: int = 8
    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """Policy/model choices for one experiment arm."""

    policy_kind: PolicyKind = "covla"
    tiny_vla_backbone: str = "vla-adapter-qwen2.5-0.5b"
    large_vla_backbone: str = "openvla-oft-7b"
    freeze_vla_backbone: bool = True
    train_adapter: bool = True
    train_communication: bool = True
    train_assignment: bool = True
    train_critic: bool = True
    communication_mode: CommunicationMode = "latent"
    assignment_mode: AssignmentMode = "learned"
    shared_actor_weights: bool = True
    message_tokens: int = 8
    message_dim: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings for the MARL fine-tuning stage."""

    algorithm: str = "mappo"
    total_updates: int = 1000
    rollout_episodes_per_update: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-5
    clip_range: float = 0.1
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    bc_kl_coef: float = 0.05
    dense_subgoal_reward: bool = True
    specialization_reward: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and reporting settings."""

    eval_episodes: int = 100
    report_latency: bool = True
    report_memory: bool = True
    report_bandwidth: bool = True
    report_collisions: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment-arm configuration."""

    name: str = "covla_full"
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def baseline_sweep(num_agents: int = 2, seed: int = 0) -> list[ExperimentConfig]:
    """Return the core baseline/ablation suite needed for the paper.

    The suite directly matches the paper's fairness requirements:
    - compare against a large VLA controlling the same number of robots;
    - isolate communication mode;
    - isolate learned assignment and centralized-critic training;
    - include a non-VLA MARL baseline.
    """

    task = TaskConfig(num_agents=num_agents, seed=seed)

    def cfg(
        name: str,
        policy_kind: PolicyKind,
        communication_mode: CommunicationMode,
        assignment_mode: AssignmentMode,
        *,
        train_critic: bool = True,
        train_assignment: bool = True,
        train_communication: bool = True,
        specialization_reward: bool = True,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            name=name,
            task=task,
            model=ModelConfig(
                policy_kind=policy_kind,
                communication_mode=communication_mode,
                assignment_mode=assignment_mode,
                train_critic=train_critic,
                train_assignment=train_assignment,
                train_communication=train_communication,
            ),
            training=TrainingConfig(
                specialization_reward=specialization_reward,
            ),
        )

    return [
        cfg("single_tiny_vla", "tiny_vla", "none", "single", train_critic=False, train_assignment=False),
        cfg("single_large_vla", "large_vla", "none", "single", train_critic=False, train_assignment=False),
        cfg("large_vla_centralized", "large_vla_centralized", "raw", "learned"),
        cfg("multi_tiny_no_comm", "tiny_vla", "none", "rule", train_communication=False),
        cfg("multi_tiny_text_comm", "tiny_vla", "text", "rule"),
        cfg("multi_tiny_raw_sharing", "tiny_vla", "raw", "rule"),
        cfg("multi_tiny_latent_comm", "tiny_vla", "latent", "rule"),
        cfg("mappo_visual", "mappo_visual", "vector", "learned"),
        cfg("covla_full", "covla", "latent", "learned"),
        cfg("covla_no_critic", "covla", "latent", "learned", train_critic=False),
        cfg("covla_no_learned_assignment", "covla", "latent", "rule", train_assignment=False),
        cfg("covla_no_specialization", "covla", "latent", "learned", specialization_reward=False),
    ]
