# Multi-Agent AIOps Platform

A multi-agent collaborative platform for autonomous IT operations, built on top of [AIOpsLab](https://github.com/microsoft/AIOpsLab) benchmark.

The system leverages three core agents — **Observer**, **Probe**, and **Executor** — coordinated with a **Context Compressor**, to perform intelligent monitoring, fault diagnosis, and automated remediation.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Observer Agent                  │
│   (Task planning, decision, coordination)   │
├──────────────┬──────────────────────────────┤
│  Probe Agent │        Executor Agent        │
│  (Read-only  │   (System modification &     │
│   diagnosis) │        remediation)          │
├──────────────┴──────────────────────────────┤
│          Context Compressor (LLM)           │
│   (Intelligent context summarization)       │
├─────────────────────────────────────────────┤
│         AIOpsLab Environment                │
│   (Microservice deployment, fault injection,│
│    telemetry, workload generation)          │
└─────────────────────────────────────────────┘
```

## Prerequisites

- Python >= 3.11
- [AIOpsLab](https://github.com/microsoft/AIOpsLab) dependencies (Helm, Kind, etc.)
- An LLM API key (OpenAI, OpenRouter, or compatible)

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/OpenEdgeHQ/aoi.git
cd aoi

# Install dependencies
pip install -r requirements.txt

# Install AIOpsLab
cd AIOpsLab
pip install -e .
cd ..
```

## Configuration

1. **Set up environment variables**: Copy and edit the example file:

```bash
cp .env.example .env
```

Edit `.env` with your API key and model settings:

```bash
API_SOURCE=openrouter          # or "openai"
API_KEY=your_api_key_here
API_BASE=https://openrouter.ai/api/v1
MODEL=anthropic/claude-sonnet-4.5
```

2. **Set up AIOpsLab cluster**: Follow the [AIOpsLab Quick Start](https://github.com/microsoft/AIOpsLab#-quick-start) to create a Kind cluster:

```bash
kind create cluster --config AIOpsLab/kind/kind-config-x86.yaml
```

3. **Configure AIOpsLab**:

```bash
cd AIOpsLab/aiopslab
cp config.yml.example config.yml
# Edit config.yml: set k8s_host to "kind" for local clusters
cd ../..
```

4. **(Optional) Docker Hub credentials** for pulling application images:

```bash
export DOCKER_USER="your_username"
export DOCKER_PASS="your_password"
export DOCKER_EMAIL="your_email"
```

## Usage

### Quick Start

Start the environment server and run evaluation:

```bash
# Start the AIOpsLab environment server
./start_all.sh
```

Or step by step:

```bash
# 1. Start environment server
python -m environment.aiopslab_server

# 2. Run evaluation on all tasks
python -m main_aiopslab
```

### Run a Single Task

```bash
python -m main_aiopslab --problem k8s_target_port-misconfig-detection-1
```

### Configuration Options

Key settings in `llm_config.py` (or via environment variables):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL` | LLM model name | `anthropic/claude-sonnet-4.5` |
| `TEMPERATURE` | Sampling temperature | `0.7` |
| `MAX_CONTEXT_TOKENS` | Max context tokens for Observer | `35000` |
| `MAX_OUTPUT_TOKENS` | Max LLM output tokens | `5000` |
| `DISABLE_THINKING` | Disable thinking mode (for Qwen3) | `True` |

## GRPO Training (Self-Evolve)

The system supports **Group Relative Policy Optimization (GRPO)** for training the Evolver agent to generate diverse and high-quality fault scenarios. GRPO is a reinforcement learning algorithm that:

1. Generates multiple candidate scenarios (a "group") per seed prompt
2. Uses a SOTA LLM (e.g., Claude Sonnet) as a reward model to score each candidate across multiple dimensions
3. Computes group-relative advantages within each group
4. Updates the policy model using these advantages

### Architecture

The GRPO training pipeline consists of two components:

- **Evolver** (`grpo/evolver/`): Generates diverse Kubernetes fault scenarios from seed data using an open-source policy model (e.g., Qwen2.5-7B). Trained via GRPO to maximize reward.
- **Observer** (`grpo/observer/`): Shares the same GRPO framework for training the observer agent.

### Prerequisites

```bash
# Core dependencies
pip install torch transformers datasets accelerate peft

# For TRL-based training (recommended)
pip install trl>=0.12.0

# Optional: vLLM for fast inference
pip install vllm>=0.6.0

# Optional: TensorBoard for monitoring
pip install tensorboard
```

### Datasets & Models

We provide pre-built training datasets and LoRA checkpoints on Hugging Face:

**Datasets:**

| Dataset | Description | Link |
|---------|-------------|------|
| aoi-planner-seeds-sonnet | Evolver seed scenarios (ground truth from Claude Sonnet) | [HuggingFace](https://huggingface.co/datasets/spacezenmasterr/aoi-planner-seeds-sonnet) |
| aoi-observer-training-data | Observer GRPO training data | [HuggingFace](https://huggingface.co/datasets/spacezenmasterr/aoi-observer-training-data) |

**Trained Models (LoRA Checkpoints):**

| Model | Description | Link |
|-------|-------------|------|
| aoi-evolver-lora-ckpt490 | Evolver LoRA adapter (checkpoint 490) | [HuggingFace](https://huggingface.co/spacezenmasterr/aoi-evolver-lora-ckpt490) |
| aoi-observer-lora-ckpt200 | Observer LoRA adapter (checkpoint 200) | [HuggingFace](https://huggingface.co/spacezenmasterr/aoi-observer-lora-ckpt200) |

### Prepare Seed Data

Seed data consists of JSON files from successful task evaluations (ground truth). Each seed contains `task_info`, `commands`, and `evaluation_results`. You can download the pre-built seed data from the datasets above, or generate your own by running evaluations.

```bash
# Seed data directory structure
data/gt/gt_c/claude-sonnet-4.5/
├── k8s_target_port-misconfig-detection-1.json
├── k8s_network_delay-localization-2.json
└── ...
```

### Training with TRL (Recommended)

The TRL-based trainer (`train_grpo_trl.py`) provides efficient training with optional vLLM acceleration:

```bash
# Single GPU training
python grpo/evolver/train_grpo_trl.py \
    --seed-dir data/gt/gt_c/claude-sonnet-4.5 \
    --model Qwen/Qwen3-14B \
    --reward-model anthropic/claude-sonnet-4.5 \
    --batch-size 2 \
    --num-generations 4 \
    --num-epochs 3

# Multi-GPU training with vLLM acceleration
accelerate launch --num_processes 3 --main_process_port 29500 \
    grpo/evolver/train_grpo_trl.py \
    --seed-dir data/gt/gt_c/claude-sonnet-4.5 \
    --model Qwen/Qwen3-14B \
    --use-vllm \
    --batch-size 4

# Resume from checkpoint
python grpo/evolver/train_grpo_trl.py \
    --seed-dir data/gt/gt_c/claude-sonnet-4.5 \
    --model Qwen/Qwen3-14B \
    --resume-from-checkpoint checkpoint-50

# Load weights from previous checkpoint (fresh training with new LR)
python grpo/evolver/train_grpo_trl.py \
    --seed-dir data/gt/gt_c/claude-sonnet-4.5 \
    --model Qwen/Qwen3-14B \
    --load-weights-from checkpoint-50
```

### Training with Custom Trainer

The custom GRPO trainer (`train_grpo.py`) provides more fine-grained control:

```bash
python grpo/evolver/train_grpo.py \
    --seed-dir data/gt/gt_c/claude-sonnet-4.5 \
    --policy-model Qwen/Qwen2.5-7B-Instruct \
    --reward-model anthropic/claude-sonnet-4-20250514 \
    --group-size 4 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --num-epochs 3 \
    --use-lora \
    --lora-rank 64
```

### Multi-Dimensional Reward

The reward model evaluates each generated scenario across multiple dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `solution_effectiveness` | 0.30 | Does the solution actually fix the problem? |
| `commands_completeness` | 0.20 | Are all diagnostic + resolution steps included? |
| `diversity` | 0.20 | Different from seed (anti-plagiarism) |
| `problem_validity` | 0.10 | Is the fault scenario realistic? |
| `commands_correctness` | 0.10 | Are commands syntactically correct? |
| `format` | 0.10 | JSON structure correctness |

### Monitoring

```bash
# TensorBoard
tensorboard --logdir ./logs/evolver_grpo/

# High-score candidates (>= 0.8) are automatically saved to:
# ./data/gt/grpo_training_high_score/
```

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Policy model to train | `Qwen/Qwen3-14B` |
| `--reward-model` | SOTA LLM for scoring | `anthropic/claude-sonnet-4.5` |
| `--num-generations` | Group size (candidates per seed) | `4` |
| `--batch-size` | Per-device batch size | `2` |
| `--learning-rate` | Learning rate | `1e-5` |
| `--use-lora` | Enable LoRA fine-tuning | `True` |
| `--lora-rank` | LoRA rank | `64` |
| `--use-vllm` | Enable vLLM acceleration | `False` |
| `--multi-dim-reward` | Use multi-dimensional rewards | `True` |

## Project Structure

```
.
├── agents/                  # Agent implementations
│   ├── observer_agent.py    #   Observer: task planning & coordination
│   ├── probe_agent.py       #   Probe: read-only system diagnosis
│   ├── executor_agent.py    #   Executor: system modification & repair
│   ├── compressor_agent.py  #   Context Compressor (LLM-based)
│   └── file_reader_agent.py #   File reader for metrics/traces
├── grpo/                    # GRPO training modules
│   ├── evolver/             #   Evolver agent GRPO training
│   │   ├── llm_evolver.py   #     LLM-based scenario generator
│   │   ├── evolver_config.py#     Evolver configuration
│   │   ├── grpo_trainer.py  #     Custom GRPO trainer
│   │   ├── grpo_config.py   #     GRPO training configuration
│   │   ├── reward_model.py  #     Multi-dimensional reward model
│   │   ├── data_loader.py   #     Seed data loading utilities
│   │   ├── train_grpo.py    #     Training script (custom)
│   │   ├── train_grpo_trl.py#     Training script (TRL-based)
│   │   └── prompts/         #     Prompt templates (YAML)
│   └── observer/            #   Observer agent GRPO training
├── environment/             # AIOpsLab client/server interface
├── memory/                  # Memory management module
├── prompts/                 # Agent prompt templates (YAML)
├── utils/                   # Utility functions
├── llm_config.py            # LLM configuration
├── main.py                  # Core platform logic
├── main_aiopslab.py         # Evaluation entry point
├── val_aoi.py               # Result validation
└── AIOpsLab/                # Benchmark framework (git submodule)
```

## Acknowledgements

This project is built on top of [AIOpsLab](https://github.com/microsoft/AIOpsLab) benchmark framework.
