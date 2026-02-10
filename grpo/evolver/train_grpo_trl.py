#!/usr/bin/env python3
"""
GRPO Training using TRL (Transformers Reinforcement Learning)

This is a more efficient implementation using TRL with optional vLLM acceleration.

Usage:
    # Multi-GPU training with vLLM (colocate mode - vLLM shares GPU with training)
    accelerate launch --num_processes 3 --main_process_port 29500 \\
        train_grpo_trl.py --model Qwen/Qwen3-14B --use-vllm --batch-size 4

    # Single GPU training with vLLM
    python train_grpo_trl.py --model Qwen/Qwen3-14B --use-vllm --batch-size 2
    
    # Single GPU training without vLLM
    python train_grpo_trl.py --model Qwen/Qwen3-14B --batch-size 4

Note: vLLM 0.11.x + TRL 0.26.x uses "colocate" mode where vLLM shares GPU with training.
      The "server" mode is not compatible with this version combination.
    
Requirements:
    pip install trl>=0.11.0 peft accelerate
    pip install vllm>=0.6.0  # Optional, for vLLM acceleration
"""

import os

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import argparse
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback

# Levenshtein no longer needed - using set intersection for diversity
# TRL imports - handle different versions
TRL_AVAILABLE = False
VLLM_AVAILABLE = False

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"TRL GRPO not available: {e}")
    print("Trying alternative import...")
    try:
        # Try importing without vLLM dependency
        from trl import PPOConfig, PPOTrainer
        print("PPO available, but GRPO requires vLLM fix. Use --no-vllm or fix vLLM version.")
    except ImportError:
        print("TRL not available. Install with: pip install trl>=0.11.0")

# Check vLLM separately
try:
    import vllm
    from vllm.sampling_params import SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available. Running without vLLM acceleration.")

# PEFT for LoRA
try:
    from peft import LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evolver.data_loader import load_seeds_from_directory, create_grpo_prompt, SeedScenario
from evolver.reward_model import RewardModel
from evolver.evolver_config import EvolverConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noise from httpx (HTTP Request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def create_dataset_from_seeds(seed_dir: str, max_seeds: Optional[int] = None) -> Dataset:
    """Create a HuggingFace Dataset from seed scenarios."""
    seeds = load_seeds_from_directory(
        seed_dir, 
        max_seeds=max_seeds, 
        shuffle=True, 
        filter_successful=True
    )
    
    # Fault dimensions for diversity
    fault_dimensions = [
        "network_delay", "network_loss", "disk_io", "memory_pressure",
        "cpu_throttle", "pod_failure", "service_misconfiguration",
        "authentication_issue", "database_connection", "resource_exhaustion",
    ]
    
    data = []
    for seed in seeds:
        for fault_dim in fault_dimensions[:4]:  # Use 4 fault dims per seed
            prompt = create_grpo_prompt(seed, fault_dim)
            # For Qwen3: add /no_think suffix to disable thinking mode
            # This tells the model to respond directly without internal reasoning
            prompt = prompt + "\n\n/no_think"
            data.append({
                "prompt": prompt,
                "seed_id": seed.problem_id,
                "fault_dimension": fault_dim,
            })
    
    logger.info(f"Created dataset with {len(data)} prompts from {len(seeds)} seeds")
    return Dataset.from_list(data)


class LogConfig:
    """Global configuration for console logging verbosity."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.level = "normal"  # quiet, normal, verbose
            cls._instance.verbose_steps = 5  # Print detailed logs every N steps
            cls._instance._current_step = 0
        return cls._instance
    
    def set_level(self, level: str):
        self.level = level
    
    def set_verbose_steps(self, steps: int):
        self.verbose_steps = steps
    
    def increment_step(self):
        self._current_step += 1
    
    def get_step(self) -> int:
        return self._current_step
    
    def should_log_verbose(self) -> bool:
        """Check if we should print verbose logs at current step."""
        if self.level == "quiet":
            return False
        if self.level == "verbose":
            return True
        # normal mode: log every verbose_steps
        return self._current_step % self.verbose_steps == 0
    
    def is_quiet(self) -> bool:
        return self.level == "quiet"
    
    def is_normal(self) -> bool:
        return self.level == "normal"
    
    def is_verbose(self) -> bool:
        return self.level == "verbose"


# Global log config instance
log_config = LogConfig()


class HighScoreSaver:
    """
    Save high-score candidates during training for later use.
    
    Saves candidates with total_reward >= threshold to JSON files.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._output_dir = None
            cls._instance._save_count = 0
            cls._instance._threshold = 0.8
        return cls._instance
    
    def set_output_dir(self, output_dir: str):
        """Set output directory for saving high-score data."""
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"High-score saver initialized: {self._output_dir}")
    
    def set_threshold(self, threshold: float):
        """Set score threshold for saving."""
        self._threshold = threshold
    
    def save_candidate(
        self,
        completion: str,
        prompt: str,
        total_reward: float,
        dim_scores: Dict[str, float],
        step: int,
        idx: int,
    ):
        """
        Save a high-score candidate to a JSON file.
        
        Args:
            completion: Generated text
            prompt: Input prompt
            total_reward: Total reward score
            dim_scores: Individual dimension scores
            step: Training step
            idx: Candidate index in batch
        """
        if self._output_dir is None:
            return
        
        if total_reward < self._threshold:
            return
        
        # Parse JSON from completion
        parsed = self._parse_json(completion)
        if parsed is None:
            return
        
        self._save_count += 1
        
        # Create output data
        output_data = {
            **parsed,
            "_metadata": {
                "training_step": step,
                "batch_idx": idx,
                "total_reward": total_reward,
                "dim_scores": dim_scores,
                "saved_at": datetime.now().isoformat(),
            },
        }
        
        # Generate filename
        filename = f"step{step:05d}_idx{idx}_score{total_reward:.2f}.json"
        filepath = self._output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  💾 Saved high-score candidate: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save high-score candidate: {e}")
    
    def get_save_count(self) -> int:
        """Get total number of saved candidates."""
        return self._save_count
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from completion text."""
        if not text:
            return None
        
        # Try to find JSON in code blocks first (4 backticks have priority)
        if "````json" in text:
            start = text.find("````json") + 8
            end = text.find("````", start)
            if end > start:
                text = text[start:end].strip()
        elif "````" in text:
            start = text.find("````") + 4
            end = text.find("````", start)
            if end > start:
                text = text[start:end].strip()
        elif "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        # Find JSON object
        brace_start = text.find("{")
        if brace_start < 0:
            return None
        
        depth = 0
        json_end = -1
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
        
        if json_end < 0:
            return None
        
        json_text = text[brace_start:json_end]
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None


# Global instance
high_score_saver = HighScoreSaver()


class CompletionStatsTracker:
    """
    Track per-completion statistics for compact logging in normal mode.
    Shows: command count and total reward score for each candidate.
    """
    _instance = None
    
    # High score threshold for tracking
    HIGH_SCORE_THRESHOLD = 0.8
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._completions = {}  # idx -> {cmd_count, total_reward, dim_scores, completion, prompt}
            cls._instance._batch_count = 0
        return cls._instance
    
    def start_batch(self):
        """Reset for new batch."""
        self._completions.clear()
        self._batch_count += 1
    
    def record_completion(self, idx: int, completion: str, prompt: str = ""):
        """Record completion text and prompt for saving high-score candidates."""
        if idx not in self._completions:
            self._completions[idx] = {
                "cmd_count": 0, "total_reward": 0.0, "dim_scores": {},
                "completion": completion, "prompt": prompt
            }
        else:
            self._completions[idx]["completion"] = completion
            self._completions[idx]["prompt"] = prompt
    
    def record_commands(self, idx: int, cmd_count: int):
        """Record command count for a completion."""
        if idx not in self._completions:
            self._completions[idx] = {
                "cmd_count": cmd_count, "total_reward": 0.0, "dim_scores": {},
                "completion": "", "prompt": ""
            }
        else:
            self._completions[idx]["cmd_count"] = cmd_count
    
    def record_reward(self, idx: int, dimension: str, reward: float):
        """Record reward for a dimension."""
        if idx not in self._completions:
            self._completions[idx] = {
                "cmd_count": 0, "total_reward": 0.0, "dim_scores": {},
                "completion": "", "prompt": ""
            }
        self._completions[idx]["dim_scores"][dimension] = reward
        self._completions[idx]["total_reward"] = sum(self._completions[idx]["dim_scores"].values())
    
    def get_high_score_ratio(self) -> float:
        """
        Calculate the ratio of candidates with total_reward >= HIGH_SCORE_THRESHOLD.
        
        Returns:
            Ratio (0.0 to 1.0) of high-score candidates
        """
        if not self._completions:
            return 0.0
        
        high_score_count = sum(
            1 for info in self._completions.values()
            if info["total_reward"] >= self.HIGH_SCORE_THRESHOLD
        )
        return high_score_count / len(self._completions)
    
    def get_total_rewards(self) -> List[float]:
        """Get list of all total rewards in current batch."""
        return [info["total_reward"] for info in self._completions.values()]
    
    def get_high_score_candidates(self) -> List[Dict[str, Any]]:
        """
        Get all candidates with total_reward >= HIGH_SCORE_THRESHOLD.
        
        Returns:
            List of dicts with {idx, completion, prompt, total_reward, dim_scores}
        """
        candidates = []
        for idx, info in self._completions.items():
            if info["total_reward"] >= self.HIGH_SCORE_THRESHOLD:
                candidates.append({
                    "idx": idx,
                    "completion": info.get("completion", ""),
                    "prompt": info.get("prompt", ""),
                    "total_reward": info["total_reward"],
                    "dim_scores": info["dim_scores"],
                })
        return candidates
    
    def log_batch_summary(self):
        """Print compact summary of all completions in current batch."""
        if not self._completions:
            return
        
        # Dimension abbreviations for compact display
        dim_abbrev = {
            "solution_effectiveness": "sol",
            "commands_completeness": "cmp",
            "problem_validity": "val",
            "commands_correctness": "cor",
            "format": "fmt",
            "diversity": "div",
        }
        
        # Calculate high score ratio
        high_score_ratio = self.get_high_score_ratio()
        high_score_count = sum(
            1 for info in self._completions.values()
            if info["total_reward"] >= self.HIGH_SCORE_THRESHOLD
        )
        
        logger.info(f"[Batch {self._batch_count}] Candidates ({high_score_count}/{len(self._completions)} >= {self.HIGH_SCORE_THRESHOLD}):")
        for idx in sorted(self._completions.keys()):
            info = self._completions[idx]
            cmd_count = info["cmd_count"]
            total = info["total_reward"]
            dim_scores = info["dim_scores"]
            
            # Build dimension scores string
            dim_parts = []
            for dim_name, abbr in dim_abbrev.items():
                score = dim_scores.get(dim_name, 0.0)
                # Show as raw score (already weighted, max ~0.35 for highest weight)
                dim_parts.append(f"{abbr}={score:.2f}")
            
            dim_str = " ".join(dim_parts)
            # Mark high-score candidates with ★
            marker = "★" if total >= self.HIGH_SCORE_THRESHOLD else " "
            logger.info(f"  {marker}#{idx+1}: cmds={cmd_count:2d} | {dim_str} | total={total:.3f}")


# Global instance
completion_stats = CompletionStatsTracker()


class RewardStatsTracker:
    """
    Global tracker for reward statistics across all dimensions.
    Records max, min, mean for each dimension per step.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stats = {}  # dimension -> list of rewards in current step
            cls._instance._step_stats = {}  # dimension -> {max, min, mean} for logging
        return cls._instance
    
    def record(self, dimension: str, rewards: List[float]):
        """Record rewards for a dimension in current step."""
        if dimension not in self._stats:
            self._stats[dimension] = []
        self._stats[dimension].extend(rewards)
        
        # Update step stats
        all_rewards = self._stats[dimension]
        valid_rewards = [r for r in all_rewards if r > 0]  # Exclude failures for stats
        
        if valid_rewards:
            self._step_stats[dimension] = {
                "max": max(all_rewards),
                "min": min(all_rewards),
                "mean": sum(all_rewards) / len(all_rewards),
                "max_valid": max(valid_rewards),
                "success_rate": len(valid_rewards) / len(all_rewards),
            }
        else:
            self._step_stats[dimension] = {
                "max": max(all_rewards) if all_rewards else 0.0,
                "min": min(all_rewards) if all_rewards else 0.0,
                "mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
                "max_valid": 0.0,
                "success_rate": 0.0,
            }
    
    def get_step_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current step statistics for all dimensions."""
        return self._step_stats.copy()
    
    def clear_step(self):
        """Clear stats for next step."""
        self._stats.clear()
        self._step_stats.clear()
    
    def log_and_print(self, force: bool = False):
        """Print stats to console and return for TensorBoard logging."""
        if not self._step_stats:
            return {}
        
        # Only print detailed stats in verbose mode or at verbose_steps intervals
        # In normal mode, completion_stats provides the compact summary
        if force or log_config.is_verbose() or (log_config.is_normal() and log_config.should_log_verbose()):
            logger.info("=" * 60)
            logger.info(f"REWARD STATS (step {log_config.get_step()}):")
            for dim, stats in sorted(self._step_stats.items()):
                logger.info(
                    f"  [{dim}] max={stats['max']:.4f}, max_valid={stats['max_valid']:.4f}, "
                    f"mean={stats['mean']:.4f}, success={stats['success_rate']:.1%}"
                )
            logger.info("=" * 60)
        
        return self._step_stats


# Global instance
reward_stats_tracker = RewardStatsTracker()


class RewardStatsCallback(TrainerCallback):
    """
    Callback to log reward statistics to TensorBoard after each step.
    
    Logs only essential metrics:
    - Each reward dimension (mean only)
    - Total reward (mean, max)
    - High score ratio (>= 0.8)
    - Loss (logged by trainer)
    """
    
    def __init__(self):
        super().__init__()
        self._tb_writer = None
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        # Start new batch for completion stats
        completion_stats.start_batch()
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        # Increment step counter for log config
        log_config.increment_step()
        
        # In normal mode: print compact summary (cmds count + total score)
        if log_config.is_normal():
            completion_stats.log_batch_summary()
        
        # Save high-score candidates
        high_score_candidates = completion_stats.get_high_score_candidates()
        for candidate in high_score_candidates:
            high_score_saver.save_candidate(
                completion=candidate["completion"],
                prompt=candidate["prompt"],
                total_reward=candidate["total_reward"],
                dim_scores=candidate["dim_scores"],
                step=state.global_step,
                idx=candidate["idx"],
            )
        
        # Get and log detailed stats (verbose mode or TensorBoard)
        stats = reward_stats_tracker.log_and_print()
        
        # Prepare simplified metrics for TensorBoard
        tb_metrics = {}
        
        # 1. Each reward dimension - only mean value
        if stats:
            for dim, dim_stats in stats.items():
                tb_metrics[f"reward/{dim}"] = dim_stats["mean"]
        
        # 2. Total reward statistics
        total_rewards = completion_stats.get_total_rewards()
        if total_rewards:
            tb_metrics["reward/total_mean"] = sum(total_rewards) / len(total_rewards)
            tb_metrics["reward/total_max"] = max(total_rewards)
        
        # 3. High score ratio (>= 0.8)
        high_score_ratio = completion_stats.get_high_score_ratio()
        tb_metrics["reward/high_score_ratio"] = high_score_ratio
        
        # Try to get TensorBoard writer from trainer
        if self._tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                if hasattr(args, 'logging_dir') and args.logging_dir:
                    self._tb_writer = SummaryWriter(log_dir=args.logging_dir)
                    logger.info(f"Created TensorBoard writer at: {args.logging_dir}")
            except Exception as e:
                logger.warning(f"Could not create TensorBoard writer: {e}")
        
        # Write directly to TensorBoard
        if self._tb_writer is not None and tb_metrics:
            for metric_name, value in tb_metrics.items():
                self._tb_writer.add_scalar(metric_name, value, state.global_step)
            self._tb_writer.flush()
        
        # Also store for trainer's log system (backup)
        if not hasattr(state, 'reward_stats_buffer'):
            state.reward_stats_buffer = {}
        state.reward_stats_buffer.update(tb_metrics)
        
        # Clear stats for next step
        reward_stats_tracker.clear_step()
        
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging. Add our custom metrics."""
        if logs is not None and hasattr(state, 'reward_stats_buffer'):
            logs.update(state.reward_stats_buffer)
            state.reward_stats_buffer = {}
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Close TensorBoard writer and log summary when training ends."""
        # Log high-score saver summary
        save_count = high_score_saver.get_save_count()
        if save_count > 0:
            logger.info(f"Training complete! Saved {save_count} high-score candidates (>= 0.8)")
        
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        return control


class MultiDimRewardCache:
    """
    Cache for reward model scores to avoid calling the reward model multiple times
    for the same completion when using multi-dimensional rewards.
    
    Tracks JSON parse status separately for format reward calculation.
    """
    def __init__(self, reward_model: RewardModel):
        self.reward_model = reward_model
        self._loop = None
        self._cache: Dict[str, Any] = {}  # completion_hash -> ScenarioScore
        self._json_valid: Dict[str, bool] = {}  # completion_hash -> True if JSON parsed successfully
    
    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop
    
    def is_json_valid(self, completion: str) -> bool:
        """Check if JSON was successfully parsed for this completion."""
        cache_key = hash(completion)
        return self._json_valid.get(cache_key, False)
    
    def get_score(self, completion: str) -> Optional[Any]:
        """Get cached score or compute new one."""
        cache_key = hash(completion)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Parse and score
        parsed = self._parse_json(completion)
        if parsed is None:
            if log_config.should_log_verbose():
                logger.warning(f"JSON parse failed -> other dims get 10%, format gets 0")
            self._cache[cache_key] = None
            self._json_valid[cache_key] = False
            return None
        
        # JSON parsed successfully
        self._json_valid[cache_key] = True
        
        # Add required fields
        if "fault_dimension" not in parsed:
            parsed["fault_dimension"] = "unknown"
        if "candidate_id" not in parsed:
            parsed["candidate_id"] = 0
        
        try:
            loop = self._get_loop()
            score = loop.run_until_complete(
                self.reward_model.score_single(parsed)
            )
            self._cache[cache_key] = score
            return score
        except Exception as e:
            logger.warning(f"Reward scoring failed: {e}")
            self._cache[cache_key] = None
            # Even if scoring failed, JSON was valid
            return None
    
    def clear_cache(self):
        """Clear the score cache."""
        self._cache.clear()
        self._json_valid.clear()
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse ONLY the JSON object from text (no extra content)."""
        if not text:
            return None
        
        # Try to find JSON in code blocks first (4 backticks have priority)
        if "````json" in text:
            start = text.find("````json") + 8
            end = text.find("````", start)
            if end > start:
                text = text[start:end].strip()
        elif "````" in text:
            start = text.find("````") + 4
            end = text.find("````", start)
            if end > start:
                text = text[start:end].strip()
        elif "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        # Find JSON object by matching braces
        brace_start = text.find("{")
        if brace_start < 0:
            return None
        
        # Find the matching closing brace
        depth = 0
        json_end = -1
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
        
        if json_end < 0:
            return None
        
        # Extract only the JSON portion (nothing before or after)
        json_text = text[brace_start:json_end]
        
        try:
            parsed = json.loads(json_text)
            logger.debug(f"Parsed JSON keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
            return parsed
        except json.JSONDecodeError as e:
            if log_config.should_log_verbose():
                logger.warning(f"JSON parse error: {e}")
            return None


class DiversityRewardWrapper:
    """
    Reward wrapper that penalizes outputs too similar to the seed.
    
    Compares generated commands with seed commands from the prompt.
    Higher similarity = Lower reward (to prevent "copying homework").
    """
    
    __name__ = "reward_diversity"
    
    # Fallback ratio when JSON parse fails (preserve gradient for learning)
    JSON_FAIL_FALLBACK_RATIO = 0.10  # 10% of max score
    
    def __init__(self, weight: float = 0.15):
        self.weight = weight
    
    def __call__(self, completions: List[str] = None, prompts: List[str] = None, **kwargs) -> List[float]:
        """Calculate diversity reward based on command similarity with seed."""
        if completions is None:
            completions = kwargs.get("completions", [])
        if prompts is None:
            prompts = kwargs.get("prompts", [])
        
        rewards = []
        for idx, completion in enumerate(completions):
            if isinstance(completion, (list, tuple)):
                rewards.append(0.0)
                continue
            
            completion_str = str(completion)
            
            # Get corresponding prompt (if available)
            prompt_str = str(prompts[idx]) if idx < len(prompts) else ""
            
            # Extract seed commands from prompt
            seed_commands = self._extract_seed_commands(prompt_str)
            
            # Extract generated commands from completion
            generated_commands = self._extract_generated_commands(completion_str)
            
            if not seed_commands or not generated_commands:
                # Can't compare -> give 10% fallback (preserve gradient for learning)
                fallback_reward = self.weight * self.JSON_FAIL_FALLBACK_RATIO
                if log_config.is_verbose():
                    logger.warning(f"[Diversity {idx+1}] seed_cmds={len(seed_commands)}, gen_cmds={len(generated_commands)} -> reward={fallback_reward:.3f} (parse failed, 10% fallback)")
                rewards.append(fallback_reward)
                completion_stats.record_commands(idx, len(generated_commands))
                completion_stats.record_reward(idx, "diversity", fallback_reward)
                continue
            
            # Record command count for compact logging
            completion_stats.record_commands(idx, len(generated_commands))
            
            # Calculate diversity score using set intersection
            # Returns 0-1, higher = more diverse (less similar to seed)
            diversity_score = self._calculate_similarity(seed_commands, generated_commands)
            reward = diversity_score * self.weight
            
            if log_config.is_verbose():
                logger.debug(f"[Diversity {idx+1}] seed_cmds={len(seed_commands)}, gen_cmds={len(generated_commands)}, diversity={diversity_score:.2%}, reward={reward:.3f}")
            
            rewards.append(reward)
            completion_stats.record_reward(idx, "diversity", reward)
        
        # Record stats for tracking
        reward_stats_tracker.record("diversity", rewards)
        
        # Only show detailed rewards in verbose mode
        if log_config.is_verbose():
            max_reward = max(rewards) if rewards else 0.0
            rewards_str = [f'{r:.3f}' if r > 0 else '0.000(FAIL)' for r in rewards]
            logger.info(f"[diversity] rewards: {rewards_str} | MAX={max_reward:.4f}")
        
        return rewards
    
    def _extract_seed_commands(self, prompt: str) -> List[str]:
        """Extract seed commands from prompt text."""
        commands = []
        import re
        
        # Pattern 1: exec_shell("...") format - handles both single and double quotes
        # Also handles escaped quotes inside
        exec_shell_pattern = r'exec_shell\s*\(\s*"([^"\\]*(?:\\.[^"\\]*)*)"\s*\)'
        for match in re.finditer(exec_shell_pattern, prompt):
            cmd = match.group(1).strip()
            # Unescape any escaped characters
            cmd = cmd.replace('\\"', '"').replace("\\'", "'")
            cmd = ' '.join(cmd.split())
            if cmd and len(cmd) > 5:  # Skip empty or too short commands
                commands.append(cmd)
        
        # Pattern 1b: exec_shell with single quotes
        exec_shell_single_pattern = r"exec_shell\s*\(\s*'([^'\\]*(?:\\.[^'\\]*)*)'\s*\)"
        for match in re.finditer(exec_shell_single_pattern, prompt):
            cmd = match.group(1).strip()
            cmd = cmd.replace("\\'", "'").replace('\\"', '"')
            cmd = ' '.join(cmd.split())
            if cmd and len(cmd) > 5:
                commands.append(cmd)
        
        # Pattern 2: Numbered commands "1. kubectl ..." or "- kubectl ..."
        if not commands:
            line_pattern = r'^\s*(?:\d+\.|-)\s*(kubectl\s+.+|curl\s+.+|docker\s+.+|helm\s+.+)$'
            for line in prompt.split('\n'):
                match = re.match(line_pattern, line, re.IGNORECASE)
                if match:
                    cmd = match.group(1).strip()
                    cmd = ' '.join(cmd.split())
                    if len(cmd) > 5:
                        commands.append(cmd)
        
        # Pattern 3: Direct kubectl/curl commands in quotes (fallback)
        if not commands:
            quote_pattern = r'["\']([^"\']*(?:kubectl|curl|docker|helm)[^"\']*)["\']'
            for match in re.finditer(quote_pattern, prompt):
                cmd = match.group(1).strip()
                if cmd.startswith(('kubectl', 'curl', 'docker', 'helm')) and len(cmd) > 10:
                    cmd = ' '.join(cmd.split())
                    commands.append(cmd)
        
        # Deduplicate while preserving order
        seen = set()
        unique_commands = []
        for cmd in commands:
            if cmd not in seen:
                seen.add(cmd)
                unique_commands.append(cmd)
        
        logger.debug(f"Extracted {len(unique_commands)} seed commands from prompt")
        return unique_commands
    
    def _extract_generated_commands(self, completion: str) -> List[str]:
        """Extract commands from generated JSON."""
        import re
        
        try:
            # Parse JSON (4 backticks have priority)
            if "````json" in completion:
                start = completion.find("````json") + 8
                end = completion.find("````", start)
                if end > start:
                    completion = completion[start:end].strip()
            elif "````" in completion:
                start = completion.find("````") + 4
                end = completion.find("````", start)
                if end > start:
                    completion = completion[start:end].strip()
            elif "```json" in completion:
                start = completion.find("```json") + 7
                end = completion.find("```", start)
                if end > start:
                    completion = completion[start:end].strip()
            elif "```" in completion:
                start = completion.find("```") + 3
                end = completion.find("```", start)
                if end > start:
                    completion = completion[start:end].strip()
            
            brace_start = completion.find("{")
            if brace_start >= 0:
                completion = completion[brace_start:]
            
            data = json.loads(completion)
            commands = data.get("commands", [])
            
            # Normalize commands - handle both formats:
            # 1. "kubectl get pods ..."
            # 2. "exec_shell(\"kubectl get pods ...\")"
            normalized = []
            exec_shell_pattern = r'exec_shell\s*\(\s*["\'](.+?)["\']\s*\)'
            
            for cmd in commands:
                if isinstance(cmd, str):
                    # Check if it's exec_shell format
                    match = re.match(exec_shell_pattern, cmd)
                    if match:
                        cmd = match.group(1)
                    # Remove extra spaces and normalize
                    cmd = ' '.join(cmd.split())
                    normalized.append(cmd)
            
            return normalized
        except:
            return []
    
    def _calculate_similarity(self, seed_cmds: List[str], gen_cmds: List[str]) -> float:
        """
        Calculate diversity score using set intersection ratio.
        
        Formula: intersection_ratio = len(set(gen_cmds) & set(seed_cmds)) / len(gen_cmds)
        
        This measures what percentage of generated commands are exact copies from seed.
        
        - intersection_ratio > 0.6: Penalty (-1.0) for copying too much
        - intersection_ratio <= 0.6: Reward (1 - ratio), higher diversity = higher reward
        """
        # Threshold for "too much copying"
        INTERSECTION_THRESHOLD = 0.6
        
        # Normalize commands for comparison
        seed_set = set(cmd.lower().strip() for cmd in seed_cmds)
        gen_set = set(cmd.lower().strip() for cmd in gen_cmds)
        
        # Calculate intersection ratio
        intersection = seed_set & gen_set
        intersection_ratio = len(intersection) / len(gen_cmds) if gen_cmds else 0.0
        
        # Calculate diversity score
        if intersection_ratio > INTERSECTION_THRESHOLD:
            # Too much copying -> strong penalty
            diversity_score = -1.0
        else:
            # Normal case: reward = 1 - ratio
            # 0% overlap -> score = 1.0 (max reward)
            # 60% overlap -> score = 0.4 (threshold)
            diversity_score = 1.0 - intersection_ratio
        
        logger.debug(f"Diversity: intersection={len(intersection)}/{len(gen_cmds)} ({intersection_ratio:.1%}), score={diversity_score:.3f}")
        
        return diversity_score

class FormatRewardWrapper:
    """
    Reward wrapper for format checking - based purely on JSON parsing success.
    
    - JSON parsed successfully -> Full score (weight * 1.0)
    - JSON parse failed -> 0.0
    
    This is determined by code, NOT by the reward model LLM.
    """
    
    __name__ = "reward_format"
    
    def __init__(self, cache: MultiDimRewardCache, weight: float = 0.05):
        self.cache = cache
        self.weight = weight
    
    def __call__(self, completions: List[str] = None, prompts: List[str] = None, **kwargs) -> List[float]:
        """Score completions based on JSON validity."""
        if completions is None:
            completions = kwargs.get("completions", [])
        
        rewards = []
        for idx, completion in enumerate(completions):
            if isinstance(completion, (list, tuple)):
                rewards.append(0.0)
                completion_stats.record_reward(idx, "format", 0.0)
                continue
            
            completion_str = str(completion)
            
            # Trigger caching (will also check JSON validity)
            self.cache.get_score(completion_str)
            
            # Check if JSON was valid
            is_valid = self.cache.is_json_valid(completion_str)
            
            # Full score if valid, 0 if invalid
            reward = self.weight if is_valid else 0.0
            rewards.append(reward)
            completion_stats.record_reward(idx, "format", reward)
        
        # Record stats for tracking
        reward_stats_tracker.record("format", rewards)
        
        if log_config.is_verbose():
            max_reward = max(rewards) if rewards else 0.0
            rewards_str = [f'{r:.3f}' if r > 0 else '0.000(FAIL)' for r in rewards]
            logger.info(f"[format] rewards: {rewards_str} | MAX={max_reward:.4f}")
        
        return rewards


class DimensionRewardWrapper:
    """
    Reward wrapper for a specific dimension.
    
    Each dimension has its own reward function that extracts the specific score
    from the cached multi-dimensional reward model output.
    
    When JSON parsing fails:
    - Give 10% of max score (to preserve learning gradient)
    - This is better than 0 because the model at least tried
    
    When JSON is valid but reward model failed:
    - Give 0 (no fallback)
    """
    
    # Fallback ratio when JSON parse fails (preserve gradient for learning)
    JSON_FAIL_FALLBACK_RATIO = 0.10  # 10% of max score
    
    # Track which dimensions should record completions (only first one)
    _first_dimension = None
    
    def __init__(self, cache: MultiDimRewardCache, dimension: str, weight: float = 1.0, record_completions: bool = False):
        self.cache = cache
        self.dimension = dimension
        self.weight = weight
        self.record_completions = record_completions  # Only first dimension should record
        self.__name__ = f"reward_{dimension}"  # TRL requires __name__
    
    def __call__(self, completions: List[str] = None, prompts: List[str] = None, **kwargs) -> List[float]:
        """Score completions for this specific dimension."""
        if completions is None:
            completions = kwargs.get("completions", [])
        if prompts is None:
            prompts = kwargs.get("prompts", [])
        
        rewards = []
        for idx, completion in enumerate(completions):
            if isinstance(completion, (list, tuple)):
                rewards.append(0.0)
                completion_stats.record_reward(idx, self.dimension, 0.0)
                continue
            
            completion_str = str(completion)
            
            # Record completion text for high-score saving (only first dimension does this)
            if self.record_completions:
                prompt_str = str(prompts[idx]) if idx < len(prompts) else ""
                completion_stats.record_completion(idx, completion_str, prompt_str)
            
            score = self.cache.get_score(completion_str)
            
            if score is None:
                # Check if it was a JSON parse failure or scoring failure
                is_json_valid = self.cache.is_json_valid(completion_str)
                
                if not is_json_valid:
                    # JSON parse failed -> give 10% of max score (preserve gradient)
                    fallback_reward = self.weight * self.JSON_FAIL_FALLBACK_RATIO
                    logger.debug(f"[{self.dimension}][{idx}] JSON invalid -> fallback reward={fallback_reward:.3f}")
                    rewards.append(fallback_reward)
                    completion_stats.record_reward(idx, self.dimension, fallback_reward)
                else:
                    # JSON valid but scoring failed -> 0
                    logger.debug(f"[{self.dimension}][{idx}] scoring failed -> reward=0.0")
                    rewards.append(0.0)
                    completion_stats.record_reward(idx, self.dimension, 0.0)
                continue
            
            # Extract the specific dimension score (NO fallback - 0.0 if missing)
            dim_score = getattr(score, f"{self.dimension}_score", 0.0)
            
            # Normalize from 0-10 to 0-1 range and apply weight
            normalized = (dim_score / 10.0) * self.weight
            rewards.append(normalized)
            
            # Record for compact summary
            completion_stats.record_reward(idx, self.dimension, normalized)
        
        # Record stats for tracking
        reward_stats_tracker.record(self.dimension, rewards)
        
        # Only show detailed rewards in verbose mode
        if log_config.is_verbose():
            max_reward = max(rewards) if rewards else 0.0
            rewards_str = [f'{r:.3f}' if r > 0 else '0.000(FAIL)' for r in rewards]
            logger.info(f"[{self.dimension}] rewards: {rewards_str} | MAX={max_reward:.4f}")
        
        return rewards


class AsyncRewardWrapper:
    """
    Wrapper to use async reward model in sync context.
    Returns overall score (weighted average of all dimensions).
    """
    
    # TRL requires __name__ attribute for reward functions
    __name__ = "aoi_reward_overall"
    
    def __init__(self, reward_model: RewardModel):
        self.reward_model = reward_model
        self._loop = None
    
    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop
    
    def __call__(self, completions: List[str] = None, prompts: List[str] = None, **kwargs) -> List[float]:
        """Score completions and return rewards.
        
        TRL 0.18+ passes additional kwargs like completion_ids, prompt_ids.
        We ignore these and just use the completion text.
        """
        # Handle different TRL versions - completions might be passed as kwarg
        if completions is None:
            completions = kwargs.get("completions", [])
        
        # Debug: print what we received (only in verbose mode)
        if log_config.is_verbose():
            logger.info(f"Reward function called with {len(completions) if completions else 0} completions")
            if completions and len(completions) > 0:
                first_comp = str(completions[0])[:300] if completions[0] else "None"
                logger.info(f"First completion preview: {first_comp}...")
        
        rewards = []
        loop = self._get_loop()
        
        for idx, completion in enumerate(completions):
            try:
                # Handle if completion is a list of token ids (convert to string)
                if isinstance(completion, (list, tuple)):
                    rewards.append(0.0)
                    continue

                completion_str = str(completion)
                
                # Print full completion only in verbose mode
                if log_config.is_verbose():
                    logger.info(f"\n{'='*60}")
                    logger.info(f"[Completion {idx+1}/{len(completions)}] Full output:")
                    logger.info(f"{completion_str}")
                    logger.info(f"{'='*60}")

                # Parse JSON from completion text
                parsed = self._parse_json(completion_str)
                if parsed is None:
                    if log_config.is_verbose():
                        logger.warning(f"[Completion {idx+1}] Failed to parse JSON, reward=0.0")
                    rewards.append(0.0)
                    continue
                
                # Add required fields
                if "fault_dimension" not in parsed:
                    parsed["fault_dimension"] = "unknown"
                if "candidate_id" not in parsed:
                    parsed["candidate_id"] = 0
                
                # Score using reward model
                score = loop.run_until_complete(
                    self.reward_model.score_single(parsed)
                )
                
                # Print reward model score details only in verbose mode
                if log_config.is_verbose():
                    logger.info(f"[Completion {idx+1}] Reward Model Scores (Multi-Dimensional):")
                    logger.info(f"  problem_validity: {score.problem_validity_score:.2f}")
                    logger.info(f"  commands_completeness: {score.commands_completeness_score:.2f} (weight: 0.20)")
                    logger.info(f"  commands_correctness: {score.commands_correctness_score:.2f}")
                    logger.info(f"  solution_effectiveness: {score.solution_effectiveness_score:.2f} (weight: 0.30)")
                    logger.info(f"  format: {score.format_score:.2f}")
                    logger.info(f"  >>> OVERALL: {score.overall_score:.2f}")
                
                rewards.append(score.overall_score)
                
            except Exception as e:
                if log_config.is_verbose():
                    logger.warning(f"[Completion {idx+1}] Reward scoring failed: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                rewards.append(0.0)
        
        # Only show batch summary in verbose mode (normal mode uses completion_stats)
        if log_config.is_verbose():
            logger.info(f"\n[Batch Summary] Rewards: {rewards}")
            logger.info(f"[Batch Summary] Mean: {sum(rewards)/len(rewards) if rewards else 0:.4f}, Max: {max(rewards) if rewards else 0:.4f}, Min: {min(rewards) if rewards else 0:.4f}")
        
        return rewards
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse ONLY the JSON object from text (no extra content)."""
        if not text:
            return None
        
        # Try to find JSON in code blocks first (4 backticks have priority)
        if "````json" in text:
            start = text.find("````json") + 8
            end = text.find("````", start)
            if end > start:
                text = text[start:end].strip()
        elif "````" in text:
            start = text.find("````") + 4
            end = text.find("````", start)
            if end > start:
                text = text[start:end].strip()
        elif "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        # Find JSON object by matching braces
        brace_start = text.find("{")
        if brace_start < 0:
            return None
        
        # Find the matching closing brace
        depth = 0
        json_end = -1
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
        
        if json_end < 0:
            return None
        
        # Extract only the JSON portion (nothing before or after)
        json_text = text[brace_start:json_end]
        
        try:
            parsed = json.loads(json_text)
            logger.debug(f"Parsed JSON keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'}")
            return parsed
        except json.JSONDecodeError as e:
            if log_config.should_log_verbose():
                logger.warning(f"JSON parse error: {e}")
            return None


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training with TRL")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--model-path", type=str, default=None)
    
    # Reward
    parser.add_argument("--reward-model", type=str, default="anthropic/claude-sonnet-4.5")
    parser.add_argument("--reward-api-key", type=str, default=None)
    parser.add_argument("--reward-api-base", type=str, default="https://openrouter.ai/api/v1")
    
    # Data
    parser.add_argument("--seed-dir", type=str, required=True)
    parser.add_argument("--max-seeds", type=int, default=None)
    
    # Training
    parser.add_argument("--num-generations", type=int, default=4, help="Group size")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Per-device batch size (default 2 for 14B model, increase for smaller models)")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                       help="Gradient accumulation steps (increase to compensate for smaller batch_size)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                       help="Max generation length (affects VRAM usage)")

    # LoRA
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    
    # vLLM (colocate mode - vLLM shares GPU with training)
    parser.add_argument("--use-vllm", action="store_true", default=False,
                       help="Use vLLM for fast inference (colocate mode)")
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.5,
                       help="vLLM GPU memory utilization (default 0.5, shared with training)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./checkpoints/grpo_trl")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume training, including optimizer/scheduler state (e.g., checkpoint-50)")
    parser.add_argument("--load-weights-from", type=str, default=None,
                       help="Path to checkpoint to load model weights ONLY (fresh training with new LR)")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10,
                       help="Save checkpoint every N steps (default: 10)")
    parser.add_argument("--verbose-steps", type=int, default=5,
                       help="Print detailed reward logs every N steps (default: 5)")
    parser.add_argument("--log-level", type=str, default="normal",
                       choices=["quiet", "normal", "verbose"],
                       help="Console log verbosity: quiet (minimal), normal (default), verbose (all details)")
    
    # Reward mode
    parser.add_argument("--multi-dim-reward", action="store_true", default=True,
                       help="Use multi-dimensional rewards (each dimension as separate reward function)")
    parser.add_argument("--single-reward", dest="multi_dim_reward", action="store_false",
                       help="Use single overall reward (weighted average of all dimensions)")
    
    # Other
    parser.add_argument("--bf16", action="store_true", default=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize log config from args
    log_config.set_level(args.log_level)
    log_config.set_verbose_steps(args.verbose_steps)
    logger.info(f"Log level: {args.log_level}, verbose every {args.verbose_steps} steps")
    
    if not TRL_AVAILABLE:
        print("Error: TRL is not installed. Run: pip install trl>=0.12.0")
        return
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              AOI GRPO Training with TRL                       ║
║         Fast Training with vLLM Acceleration                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check distributed training status
    num_gpus = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    logger.info(f"GPU Setup: {num_gpus} GPUs available, world_size={world_size}, local_rank={local_rank}")
    
    if args.use_vllm:
        # In colocate mode (vLLM 0.11.x + TRL 0.26.x), vLLM shares GPU with training
        # This means each training process has its own vLLM instance on the same GPU
        if world_size == 1 and num_gpus > 1:
            logger.info("=" * 60)
            logger.info("SINGLE PROCESS MODE with vLLM colocate")
            logger.info(f"Available GPUs: {num_gpus}, Training on: cuda:0")
            logger.info("vLLM shares GPU memory with training (colocate mode)")
            logger.info("")
            logger.info("To use multiple GPUs with accelerate:")
            logger.info(f"  accelerate launch --num_processes {num_gpus} \\")
            logger.info(f"      {__file__} --use-vllm [other args]")
            logger.info("=" * 60)
        else:
            logger.info(f"Distributed training: {world_size} processes, vLLM in colocate mode")
    
    # Validate batch size to avoid OOM
    effective_batch = args.batch_size * args.num_generations
    logger.info(f"Effective per-step batch size: {args.batch_size} x {args.num_generations} = {effective_batch}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    model_path = args.model_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    logger.info(f"Loading seeds from {args.seed_dir}")
    dataset = create_dataset_from_seeds(args.seed_dir, args.max_seeds)
    
    # Setup reward model
    api_key = args.reward_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("Reward API key is required. Set --reward-api-key or OPENROUTER_API_KEY")
    
    evolver_config = EvolverConfig(
        api_key=api_key,
        api_base=args.reward_api_base,
        reward_model=args.reward_model,
    )
    reward_model = RewardModel(evolver_config)
    
    # Initialize high-score saver for saving candidates with score >= 0.8
    high_score_output_dir = "./data/gt/grpo_training_high_score"
    high_score_saver.set_output_dir(high_score_output_dir)
    high_score_saver.set_threshold(0.8)
    logger.info(f"High-score candidates (>= 0.8) will be saved to: {high_score_output_dir}")
    
    # Create reward function(s) based on mode
    if args.multi_dim_reward:
        # Multi-dimensional reward: each dimension is a separate reward function
        # TRL will call each function and the rewards work independently
        logger.info("Using MULTI-DIMENSIONAL rewards (each dimension separate)")
        
        # Create shared cache to avoid calling reward model multiple times
        reward_cache = MultiDimRewardCache(reward_model)
        
        # Create reward function for each dimension with its weight
        # Weights (total = 1.0):
        # - solution_effectiveness (0.30): Does the solution actually fix the problem?
        # - commands_completeness (0.20): Are all diagnostic + resolution steps included?
        # - diversity (0.20): Different from seed (anti-plagiarism)
        # - problem_validity (0.10): Is the fault scenario realistic?
        # - commands_correctness (0.10): Are commands syntactically correct?
        # - format (0.10): JSON parsing success (code-based, not LLM)
        reward_funcs = [
            # First dimension records completions for high-score saving
            DimensionRewardWrapper(reward_cache, "solution_effectiveness", weight=0.30, record_completions=True),
            DimensionRewardWrapper(reward_cache, "commands_completeness", weight=0.20),
            DimensionRewardWrapper(reward_cache, "problem_validity", weight=0.10),
            DimensionRewardWrapper(reward_cache, "commands_correctness", weight=0.10),
            # Format reward: based on JSON parsing success (not LLM evaluation)
            FormatRewardWrapper(reward_cache, weight=0.1),
            # Diversity reward: penalize copying from seed (higher similarity = lower reward)
            DiversityRewardWrapper(weight=0.20),
        ]
        
        logger.info(f"Created {len(reward_funcs)} reward functions:")
        for rf in reward_funcs:
            logger.info(f"  - {rf.__name__} (weight: {rf.weight})")
        
        reward_fn = reward_funcs  # TRL accepts list of reward functions
    else:
        # Single overall reward (weighted average)
        logger.info("Using SINGLE overall reward (weighted average of all dimensions)")
        reward_fn = AsyncRewardWrapper(reward_model)

    # LoRA config
    peft_config = None
    if args.use_lora and PEFT_AVAILABLE:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        )
        logger.info(f"Using LoRA with rank={args.lora_rank}")
    
    # Generate timestamp for TensorBoard log directory (format: YYYYMMDD_HHMM)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tensorboard_log_dir = f"./logs/evolver_grpo/{run_timestamp}"
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    
    # GRPO Config - adapt to different TRL versions
    grpo_config_kwargs = {
        "output_dir": args.output_dir,
        
        # Model init - use string dtype to avoid JSON serialization issues
        "model_init_kwargs": {
            "dtype": "bfloat16" if args.bf16 else "float16",  # TRL 0.26+ uses 'dtype' not 'torch_dtype'
            "trust_remote_code": True,
            "attn_implementation": "sdpa",  # Use PyTorch's scaled dot product attention (no flash_attn needed)
            "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
        },
        
        # Generation - TRL uses different param names in different versions
        "num_generations": args.num_generations,
        
        # Training - keep batch size small to avoid OOM
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        
        # Optimization - critical for memory
        "bf16": args.bf16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},  # More memory efficient
        
        # Logging - TensorBoard
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": ["tensorboard"],  # Enable TensorBoard logging (must be a list)
        "logging_dir": tensorboard_log_dir,  # TensorBoard log directory with timestamp
        "logging_first_step": True,  # Log the first step
        
        # vLLM acceleration
        "use_vllm": args.use_vllm,
        
        # Memory optimization
        "dataloader_num_workers": 0,  # Avoid extra memory from workers
        "dataloader_pin_memory": False,  # Save memory
    }
    
    # Check which parameters GRPOConfig supports
    import inspect
    grpo_sig = inspect.signature(GRPOConfig.__init__)
    grpo_params = list(grpo_sig.parameters.keys())
    
    # Add version-dependent parameters for vLLM
    if args.use_vllm:
        # CRITICAL: Force "colocate" mode for vLLM 0.11.x + TRL 0.26.x
        # The "server" mode's init_communicator API is not supported by vLLM 0.11.x
        # "colocate" mode runs vLLM in the same process as training, sharing GPU memory
        if "vllm_mode" in grpo_params:
            grpo_config_kwargs["vllm_mode"] = "colocate"
            logger.info("Using vLLM colocate mode (vLLM shares GPU with training)")
        
        # GPU memory for vLLM - important in colocate mode since GPU is shared
        if "vllm_gpu_memory_utilization" in grpo_params:
            grpo_config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory
        
        # Tensor parallel size
        if "vllm_tensor_parallel_size" in grpo_params:
            grpo_config_kwargs["vllm_tensor_parallel_size"] = 1
        
        # Set max_model_length to limit KV cache memory (TRL 0.26 uses 'vllm_max_model_length')
        # This is CRITICAL to avoid "KV cache memory" errors
        if "vllm_max_model_length" in grpo_params:
            grpo_config_kwargs["vllm_max_model_length"] = 8192  # Limit to 8K context
            logger.info("Setting vllm_max_model_length=8192 to limit KV cache memory")
        
        # Enforce eager mode to save memory (no CUDA graphs) - important for colocate
        if "vllm_enforce_eager" in grpo_params:
            grpo_config_kwargs["vllm_enforce_eager"] = True
        
        logger.info(f"vLLM config: mode=colocate, memory_util={args.vllm_gpu_memory}")


    if "max_completion_length" in grpo_params:
        grpo_config_kwargs["max_completion_length"] = args.max_new_tokens
    elif "response_length" in grpo_params:
        grpo_config_kwargs["response_length"] = args.max_new_tokens
    elif "max_length" in grpo_params:
        grpo_config_kwargs["max_length"] = args.max_new_tokens
    else:
        # Pass via generation_config
        logger.warning("No direct generation length param found, using default")
    
    # Note: vllm_device is only used in "server" mode
    # In "colocate" mode (which we use), vLLM shares GPU with training process
    # so vllm_device setting is ignored
    
    logger.info(f"Using TRL {getattr(__import__('trl'), '__version__', 'unknown')} API")
    logger.info(f"GRPOConfig params available: {[p for p in grpo_params if 'vllm' in p.lower() or 'max' in p.lower() or 'length' in p.lower()]}")

    grpo_config = GRPOConfig(**grpo_config_kwargs)

    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Reward Model: {args.reward_model}")
    logger.info(f"  Multi-Dim Reward: {args.multi_dim_reward}")
    logger.info(f"  Group Size: {args.num_generations}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Use vLLM: {args.use_vllm}")
    logger.info(f"  Use LoRA: {args.use_lora}")

    # Create trainer - TRL 0.18 API
    # Check which parameters GRPOTrainer accepts
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    trainer_params = list(trainer_sig.parameters.keys())

    # Handle --load-weights-from: load LoRA adapter weights only (fresh training with new LR)
    # For LoRA: we still use the base model, but load the adapter separately
    effective_model_path = model_path
    lora_weights_path = None
    
    if args.load_weights_from:
        weights_path = args.load_weights_from
        # If only checkpoint name is given (e.g., "checkpoint-10"), prepend output_dir
        if not os.path.isabs(weights_path) and not weights_path.startswith("./"):
            weights_path = os.path.join(args.output_dir, weights_path)
        
        # Check if this is a LoRA checkpoint (has adapter_config.json)
        adapter_config_path = os.path.join(weights_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            # LoRA checkpoint: use base model + load adapter
            logger.info(f"Detected LoRA checkpoint at: {weights_path}")
            logger.info(f"Will load base model from: {model_path}")
            logger.info(f"Will load LoRA adapter from: {weights_path}")
            logger.info(f"Training will start from step 0 with fresh optimizer/scheduler (LR={args.learning_rate})")
            lora_weights_path = weights_path
            # Keep using base model path
            effective_model_path = model_path
        else:
            # Full model checkpoint
            logger.info(f"Loading full model weights from: {weights_path}")
            logger.info(f"Training will start from step 0 with fresh optimizer/scheduler (LR={args.learning_rate})")
            effective_model_path = weights_path

    trainer_kwargs = {
        "model": effective_model_path,
        "args": grpo_config,
        "reward_funcs": reward_fn,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }
    
    # Add tokenizer if supported
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    
    # Clear cache before creating trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i} before trainer: allocated={mem_allocated:.2f}GB, reserved={mem_reserved:.2f}GB")
    
    trainer = GRPOTrainer(**trainer_kwargs)
    
    # Load LoRA weights from previous checkpoint if specified
    if lora_weights_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter weights from: {lora_weights_path}")
        # The model inside trainer is already a PeftModel, load the adapter weights
        trainer.model.load_adapter(lora_weights_path, adapter_name="default")
        logger.info("LoRA adapter weights loaded successfully")
    
    # Add reward stats callback for TensorBoard logging
    trainer.add_callback(RewardStatsCallback())
    logger.info("Created GRPOTrainer with RewardStatsCallback")
    
    # Log memory after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem_free = mem_total - mem_reserved
            logger.info(f"GPU {i} after model load: allocated={mem_allocated:.2f}GB, reserved={mem_reserved:.2f}GB, free={mem_free:.2f}GB")
            
            # Warn if free memory is too low
            if mem_free < 20:  # Less than 20GB free
                logger.warning(f"GPU {i} has low free memory ({mem_free:.2f}GB). Consider reducing batch_size or max_new_tokens")

    # Train
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        # If only checkpoint name is given (e.g., "checkpoint-50"), prepend output_dir
        if not os.path.isabs(checkpoint_path) and not checkpoint_path.startswith("./"):
            checkpoint_path = os.path.join(args.output_dir, checkpoint_path)
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None
        logger.info("Starting training from scratch...")
    
    try:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM Error: {e}")
        logger.error("Suggestions to fix OOM:")
        logger.error("  1. Reduce --batch-size (current: %d)", args.batch_size)
        logger.error("  2. Reduce --num-generations (current: %d)", args.num_generations)
        logger.error("  3. Reduce --max-new-tokens (current: %d)", args.max_new_tokens)
        logger.error("  4. Reduce --vllm-gpu-memory (current: %.2f)", args.vllm_gpu_memory)
        raise
    
    # Save
    trainer.save_model(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

