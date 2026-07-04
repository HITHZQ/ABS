# Tiny VLA Teams Core Experiment Scaffold

This directory contains the core experimental code scaffold for:

**Tiny VLA Teams: Efficient Multi-Agent Reinforcement Learning for Language-Conditioned Robotic Manipulation**

The code is intentionally dependency-light and currently runs on a symbolic mock task:

> Open the drawer, put the cup inside it, and place the spoon on the plate.

The mock task is not a robotics simulator. It exists to validate experiment orchestration, baseline definitions, metrics, and result logging before integrating LIBERO/robosuite and real VLA checkpoints.

## What this implements

- Core experiment configuration for the paper's required baselines.
- A runnable mock LIBERO-Coop task with:
  - sequential dependency: drawer must open before cup placement;
  - parallel subtask: spoon can be placed while drawer/cup subtasks proceed;
  - interference penalty: multiple agents entering the drawer workspace collide.
- Communication modes:
  - none;
  - text;
  - raw observation proxy;
  - learned vector proxy;
  - VLA latent-token proxy.
- Assignment modes:
  - single-agent;
  - rule-based;
  - learned-policy placeholder.
- Evaluation metrics:
  - success rate;
  - subgoal completion rates;
  - reward;
  - steps;
  - collisions;
  - communication bandwidth.

## Run the core baseline sweep

```bash
cd experiments/tiny_vla_teams
python run_core_sweep.py --num-agents 2 --eval-episodes 20
```

Output:

```text
outputs/core_sweep.jsonl
```

This corresponds to the main paper table:

- single tiny VLA;
- single large VLA;
- large VLA centralized controller with the same robot count;
- multi tiny VLA without communication;
- multi tiny VLA with text communication;
- multi tiny VLA with raw sharing;
- multi tiny VLA with latent communication;
- MAPPO visual policy;
- full CoVLA;
- key CoVLA ablations.

## Run team-size scaling

```bash
cd experiments/tiny_vla_teams
python run_team_size_sweep.py --min-agents 1 --max-agents 5 --eval-episodes 20
```

Output:

```text
outputs/team_size_sweep.jsonl
```

## Where to plug in real experiments

Replace or extend these pieces:

1. `covla_core.mock_libero_coop.MockOpenDrawerPutCupSpoonEnv`
   - Replace with a `CooperativeManipulationEnv` adapter for LIBERO/robosuite.

2. `covla_core.policies.BaselinePolicy`
   - Replace symbolic subgoal actions with Tiny VLA / OpenVLA inference.

3. `covla_core.policies.CommunicationModule`
   - Replace proxy latent payloads with real VLA-Adapter/OpenVLA intermediate tokens.

4. `covla_core.train.train_config`
   - Implement:
     - behavior cloning initialization;
     - centralized critic pretraining;
     - auxiliary communication losses;
     - MAPPO/PPO adapter fine-tuning with frozen VLA backbone.

5. `covla_core.config.baseline_sweep`
   - Keep this baseline list fixed for fairness, then add task-family variants.

## Why this design matches the paper

The scaffold forces the fairness checks needed for AAAI-style review:

- It compares against a large VLA controlling the same number of robots.
- It separates robot-count gains from communication/coordination gains.
- It logs bandwidth so latent communication can be evaluated against text and raw sharing.
- It includes no-critic, no-learned-assignment, and no-specialization ablations.
- It provides a team-size scaling entry point.
