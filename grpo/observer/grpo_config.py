# Observer GRPO Configuration
"""
Configuration for Observer GRPO training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RewardWeights:
    """
    Reward dimension weights.
    
    Normal weights (total = 1.0):
    - format: 0.10
    - summary: 0.15
    - action: 0.10
    - context_instruction: 0.30
    - context_namespace: 0.30
    - confidence: 0.05
    
    When n=1 or n=m-1, skip summary and redistribute its weight.
    """
    format: float = 0.10
    summary: float = 0.15
    action: float = 0.10
    context_instruction: float = 0.30
    context_namespace: float = 0.30
    confidence: float = 0.05
    
    def get_weights_without_summary(self) -> dict:
        """
        Get weights when summary is skipped (n=1 or n=m-1).
        Redistribute summary weight (0.15) proportionally.
        """
        remaining_total = 1.0 - self.summary
        scale = 1.0 / remaining_total
        return {
            "format": self.format * scale,
            "action": self.action * scale,
            "context_instruction": self.context_instruction * scale,
            "context_namespace": self.context_namespace * scale,
            "confidence": self.confidence * scale,
        }
    
    def get_all_weights(self) -> dict:
        """Get all weights as dict."""
        return {
            "format": self.format,
            "summary": self.summary,
            "action": self.action,
            "context_instruction": self.context_instruction,
            "context_namespace": self.context_namespace,
            "confidence": self.confidence,
        }


@dataclass
class ObserverGRPOConfig:
    """Configuration for Observer GRPO training."""
    
    # ==================== Model Configuration ====================
    # Policy model (the model to be trained)
    policy_model_path: str = "Qwen/Qwen3-14B"
    
    # Reward model (SOTA closed-source LLM for scoring)
    reward_model_api_base: str = "https://openrouter.ai/api/v1"
    reward_model_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    reward_model_name: str = "anthropic/claude-sonnet-4.5"
    reward_temperature: float = 0.0  # Low temperature for consistent scoring
    
    # Proxy configuration (set to empty string if not needed)
    proxy_url: str = ""
    retry_interval: int = 20  # Retry interval in seconds
    
    # ==================== GRPO Hyperparameters ====================
    # Group size (number of candidates per prompt)
    group_size: int = 4
    
    # KL penalty coefficient
    kl_coef: float = 0.1
    
    # Clip ratio for policy gradient
    clip_ratio: float = 0.2
    
    # Entropy bonus coefficient
    entropy_coef: float = 0.01
    
    # Whether to normalize advantages within batch
    normalize_advantages: bool = True
    
    # ==================== Generation Configuration ====================
    generation_temperature: float = 0.7
    generation_top_p: float = 0.95
    max_new_tokens: int = 4096
    do_sample: bool = True
    
    # ==================== Training Configuration ====================
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # ==================== LoRA Configuration ====================
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # ==================== Data Configuration ====================
    data_dir: str = "./data/gt/gt_observer_training_best"
    prompts_path: str = "./prompts/observer_prompts.yaml"
    reward_prompts_path: str = "./observer/prompts/reward_prompts.yaml"
    
    # ==================== Logging & Checkpointing ====================
    log_dir: str = "./logs/observer_grpo"
    checkpoint_dir: str = "./checkpoints/observer"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    
    # ==================== Hardware Configuration ====================
    device: str = "cuda"
    fp16: bool = False
    bf16: bool = True
    
    # ==================== vLLM Configuration ====================
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.5
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: int = 14000  # Max context length for vLLM
    
    # ==================== High Score Saving ====================
    high_score_dir: str = "./data/gt/observer_training_high_score"
    high_score_threshold: float = 0.8
    
    # ==================== Reward Weights ====================
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    
    # ==================== Misc ====================
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2 for GRPO")
        
        if not self.reward_model_api_key:
            self.reward_model_api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps * self.group_size


# Default configuration
DEFAULT_CONFIG = ObserverGRPOConfig()
