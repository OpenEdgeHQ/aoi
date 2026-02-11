# Evolver Configuration Module
"""
Configuration for the Evolver component in AOI Self-Evolve Training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvolverConfig:
    """Configuration for the LLM-based Evolver."""
    
    # API Configuration
    api_source: str = "openrouter"  # "openrouter" or "openai" or "anthropic"
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    api_base: str = "https://openrouter.ai/api/v1"
    
    # Model Configuration for Evolver (higher temperature for diversity)
    evolver_model: str = "anthropic/claude-sonnet-4-20250514"  # SOTA model for evolving
    evolver_temperature: float = 0.8  # Higher temperature for diverse generation
    
    # Model Configuration for Reward Model (lower temperature for consistency)
    reward_model: str = "anthropic/claude-sonnet-4-20250514"  # SOTA model for scoring
    reward_temperature: float = 0.0  # Low temperature for consistent scoring
    
    # Generation Configuration
    num_candidates: int = 8  # Number of candidates to generate per input (N)
    max_retries: int = 3  # Max retries for failed generations
    
    # Output Configuration
    output_dir: str = "./data/gt/evolver_output"
    
    # Diversity Dimensions for fault scenarios
    fault_dimensions: List[str] = field(default_factory=lambda: [
        "network_delay",
        "network_loss", 
        "disk_io",
        "memory_pressure",
        "cpu_throttle",
        "pod_failure",
        "service_misconfiguration",
        "authentication_issue",
        "database_connection",
        "resource_exhaustion",
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            # Try to get from different env vars
            self.api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
        
        if self.num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        
        if not 0.0 <= self.evolver_temperature <= 2.0:
            raise ValueError("evolver_temperature must be between 0.0 and 2.0")


@dataclass
class RewardConfig:
    """Configuration for the SOTA Reward Model."""
    
    # Scoring weights for different dimensions
    logical_consistency_weight: float = 0.35  # Weight for logical consistency scoring
    complexity_weight: float = 0.30  # Weight for complexity/value scoring
    syntax_correctness_weight: float = 0.35  # Weight for syntax correctness scoring
    
    # Score thresholds
    min_acceptable_score: float = 0.5  # Minimum score to accept a candidate
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.logical_consistency_weight + self.complexity_weight + self.syntax_correctness_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")


# Default configurations
DEFAULT_EVOLVER_CONFIG = EvolverConfig()
DEFAULT_REWARD_CONFIG = RewardConfig()
