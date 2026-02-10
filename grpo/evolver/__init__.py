# Evolver Module for AOI Self-Evolve Training
# This module implements the Evolver component for diversity evolving and RL optimization
#
# Components:
# - LLMEvolver: Generates diverse fault scenarios using closed-source LLMs
# - RewardModel: Scores generated scenarios using SOTA LLMs
# - GRPOTrainer: Trains the evolver using Group Relative Policy Optimization (requires torch)
#
# Usage:
#   # For scenario generation (no training):
#   python -m evolver.run_evolver --seed path/to/seed.json --score
#
#   # For GRPO training:
#   python -m evolver.train_grpo --seed-dir data/gt/gt_c/claude-sonnet-4.5

# Core components (always available)
from .llm_evolver import LLMEvolver, LLMEvolverSync
from .reward_model import RewardModel, RewardModelSync, ScenarioScore
from .evolver_config import EvolverConfig
from .grpo_config import GRPOConfig, RewardScoreConfig
from .data_loader import (
    SeedScenario,
    SeedDataset,
    GRPODataLoader,
    load_seed_from_file,
    load_seeds_from_directory,
)

__all__ = [
    # Evolver components
    "LLMEvolver",
    "LLMEvolverSync",
    "RewardModel",
    "RewardModelSync",
    "ScenarioScore",
    "EvolverConfig",
    
    # GRPO config (always available)
    "GRPOConfig",
    "RewardScoreConfig",
    
    # Data loading
    "SeedScenario",
    "SeedDataset",
    "GRPODataLoader",
    "load_seed_from_file",
    "load_seeds_from_directory",
]

# Optional: GRPO training (requires torch)
try:
    from .grpo_trainer import GRPOTrainer, GRPOBatchOutput
    __all__.extend(["GRPOTrainer", "GRPOBatchOutput"])
    GRPO_AVAILABLE = True
except ImportError:
    GRPO_AVAILABLE = False
    GRPOTrainer = None
    GRPOBatchOutput = None
