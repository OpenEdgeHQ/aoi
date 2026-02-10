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

## Project Structure

```
.
├── agents/                  # Agent implementations
│   ├── observer_agent.py    #   Observer: task planning & coordination
│   ├── probe_agent.py       #   Probe: read-only system diagnosis
│   ├── executor_agent.py    #   Executor: system modification & repair
│   ├── compressor_agent.py  #   Context Compressor (LLM-based)
│   └── file_reader_agent.py #   File reader for metrics/traces
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
