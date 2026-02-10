# GRPO Training Configuration
"""
Configuration for Group Relative Policy Optimization (GRPO) training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class OptimizerType(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # ==================== Model Configuration ====================
    # Policy model (the model to be trained)
    policy_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    policy_model_path: Optional[str] = None  # Local path, overrides model_name if set
    
    # Reward model (SOTA closed-source LLM for scoring)
    reward_model_api_base: str = "https://openrouter.ai/api/v1"
    reward_model_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    reward_model_name: str = "anthropic/claude-sonnet-4-20250514"
    reward_temperature: float = 0.0  # Low temperature for consistent scoring
    
    # ==================== GRPO Hyperparameters ====================
    # Group size (number of candidates per prompt)
    group_size: int = 4  # N candidates per seed
    
    # KL penalty coefficient (prevents policy from diverging too far from reference)
    kl_coef: float = 0.1
    
    # Clip ratio for policy gradient (similar to PPO clip)
    clip_ratio: float = 0.2
    
    # Entropy bonus coefficient (encourages exploration)
    entropy_coef: float = 0.01
    
    # Whether to normalize advantages within batch
    normalize_advantages: bool = True
    
    # ==================== Generation Configuration ====================
    # Temperature for generation (higher = more diverse)
    generation_temperature: float = 0.8
    
    # Top-p sampling
    generation_top_p: float = 0.95
    
    # Maximum new tokens to generate
    max_new_tokens: int = 2048
    
    # Whether to use sampling (vs greedy decoding)
    do_sample: bool = True
    
    # ==================== Training Configuration ====================
    # Learning rate
    learning_rate: float = 1e-5
    
    # Weight decay
    weight_decay: float = 0.01
    
    # Batch size (number of seeds per batch)
    batch_size: int = 2
    
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 4
    
    # Effective batch size = batch_size * gradient_accumulation_steps * group_size
    
    # Number of training epochs
    num_epochs: int = 3
    
    # Maximum training steps (overrides num_epochs if set)
    max_steps: Optional[int] = None
    
    # Warmup steps
    warmup_steps: int = 100
    
    # Optimizer type
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    
    # Scheduler type
    scheduler_type: SchedulerType = SchedulerType.COSINE
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # ==================== LoRA Configuration ====================
    # Whether to use LoRA for efficient fine-tuning
    use_lora: bool = True
    
    # LoRA rank
    lora_rank: int = 64
    
    # LoRA alpha
    lora_alpha: int = 128
    
    # LoRA dropout
    lora_dropout: float = 0.05
    
    # LoRA target modules
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # ==================== Data Configuration ====================
    # Input directory containing seed JSON files
    seed_data_dir: str = "./data/gt/gt_c/claude-sonnet-4.5"
    
    # Output directory for generated scenarios
    output_dir: str = "./data/gt/grpo_output"
    
    # Maximum number of seeds to use (None = all)
    max_seeds: Optional[int] = None
    
    # ==================== Logging & Checkpointing ====================
    # Logging directory
    log_dir: str = "./logs/grpo"
    
    # Checkpoint directory
    checkpoint_dir: str = "./checkpoints/grpo"
    
    # Logging interval (steps) - set to 1 for detailed monitoring
    logging_steps: int = 1
    
    # Evaluation interval (steps)
    eval_steps: int = 100
    
    # Checkpoint save interval (steps)
    save_steps: int = 500
    
    # Maximum checkpoints to keep
    save_total_limit: int = 3
    
    # Whether to use wandb for logging
    use_wandb: bool = False
    
    # Wandb project name
    wandb_project: str = "aoi-grpo"
    
    # ==================== Hardware Configuration ====================
    # Device
    device: str = "cuda"
    
    # Mixed precision training
    fp16: bool = False
    bf16: bool = True
    
    # Number of GPUs
    num_gpus: int = 1
    
    # DeepSpeed config path (for multi-GPU training)
    deepspeed_config: Optional[str] = None
    
    # ==================== Misc ====================
    # Random seed
    seed: int = 42
    
    # Number of workers for data loading
    num_workers: int = 4
    
    # Whether to resume from checkpoint
    resume_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        
        if not self.reward_model_api_key:
            self.reward_model_api_key = os.getenv("OPENAI_API_KEY", "")
        
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps * self.group_size


@dataclass
class RewardScoreConfig:
    """Configuration for reward scoring dimensions."""
    
    # Weight for logical consistency
    logical_consistency_weight: float = 0.35
    
    # Weight for complexity and training value
    complexity_weight: float = 0.30
    
    # Weight for syntax correctness
    syntax_correctness_weight: float = 0.35
    
    # Minimum acceptable score (0-10 scale)
    min_acceptable_score: float = 4.0
    
    # Whether to penalize repetitive content
    penalize_repetition: bool = True
    
    # Repetition penalty factor
    repetition_penalty: float = 0.1
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.logical_consistency_weight + self.complexity_weight + self.syntax_correctness_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")


# Default configurations
DEFAULT_GRPO_CONFIG = GRPOConfig()
DEFAULT_REWARD_SCORE_CONFIG = RewardScoreConfig()

