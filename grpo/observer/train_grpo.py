#!/usr/bin/env python3
"""
Observer GRPO Training using TRL + vLLM

Uses vLLM for fast inference acceleration.

Usage:
    python train_grpo.py \\
        --policy-model-path Qwen/Qwen3-14B \\
        --reward-model anthropic/claude-sonnet-4.5 \\
        --vllm-gpu-memory 0.4 \\
        --vllm-max-model-len 14000
"""

import os
# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

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

# TRL imports
TRL_AVAILABLE = False
try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError as e:
    print(f"TRL GRPO not available: {e}")

# PEFT for LoRA
try:
    from peft import LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from observer.grpo_config import ObserverGRPOConfig
from observer.data_loader import ObserverDataLoader, create_dataset_from_loader
from observer.reward_model import ObserverRewardModel, RewardScore, parse_json_output


def canonicalize_completion(completion: str) -> str:
    """
    Canonicalize model completion for reward/caching/saving.

    Training-time generations may include repeated JSON blocks or markdown fences
    (e.g., multiple ```json ... ``` sections). For scoring we only need the first
    valid JSON object. Canonicalizing improves cache hit-rate and reduces noisy
    high-score artifacts on disk.
    """
    if completion is None:
        return ""
    text = str(completion).strip()
    parsed = parse_json_output(text)
    if parsed is None:
        return text
    try:
        # Stable JSON string without markdown fences
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class LogConfig:
    """Global logging configuration."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logging_steps = 10
            cls._instance._current_step = 0
        return cls._instance
    
    def increment_step(self):
        self._current_step += 1
    
    def get_step(self) -> int:
        return self._current_step
    
    def should_log_detailed(self) -> bool:
        return self._current_step % self.logging_steps == 0


log_config = LogConfig()


class RewardScoreCache:
    """
    Cache for reward scores to avoid duplicate API calls.
    Each candidate is scored once and cached.
    """
    
    def __init__(self, reward_model: ObserverRewardModel):
        self.reward_model = reward_model
        self._cache: Dict[str, RewardScore] = {}
        self._loop = None
    
    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop
    
    def _cache_key(self, completion: str, sample_info: Dict) -> str:
        """Generate cache key from completion and sample info."""
        return f"{hash(completion)}_{sample_info.get('problem_id', '')}_{sample_info.get('current_iter', 0)}"
    
    def get_score(
        self,
        completion: str,
        sample_info: Dict[str, Any],
    ) -> RewardScore:
        """Get score from cache or compute."""
        key = self._cache_key(completion, sample_info)
        
        if key in self._cache:
            return self._cache[key]
        
        # Compute score
        loop = self._get_loop()
        score = loop.run_until_complete(
            self.reward_model.score_single(
                candidate_output=completion,
                problem_id=sample_info.get("problem_id", ""),
                task_type=sample_info.get("task_type", ""),
                task_description=sample_info.get("task_description", ""),
                current_iter=sample_info.get("current_iter", 1),
                total_iters=sample_info.get("total_iters", 1),
                expected_action=sample_info.get("expected_action", "probe"),
                execution_history=sample_info.get("execution_history", ""),
                steps=sample_info.get("steps", []),
            )
        )
        
        self._cache[key] = score
        return score
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()


class HighScoreSaver:
    """
    Save high-scoring candidates to disk for analysis.
    """
    
    def __init__(self, output_dir: str, threshold: float = 0.8):
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self._saved_count = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"HighScoreSaver initialized: dir={output_dir}, threshold={threshold}")
    
    def save_candidate(
        self,
        completion: str,
        sample_info: Dict,
        total_reward: float,
        dim_scores: Dict[str, float],
    ):
        """Save a high-scoring candidate."""
        if total_reward < self.threshold:
            return
        
        self._saved_count += 1
        
        # Create filename
        problem_id = sample_info.get("problem_id", "unknown")
        current_iter = sample_info.get("current_iter", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem_id}_iter{current_iter}_{timestamp}_{self._saved_count}.json"
        
        # Prepare data
        data = {
            "problem_id": problem_id,
            "file_path": sample_info.get("file_path", ""),
            "task_type": sample_info.get("task_type", ""),
            "current_iter": current_iter,
            "total_iters": sample_info.get("total_iters", 0),
            "expected_action": sample_info.get("expected_action", ""),
            "total_reward": total_reward,
            "dim_scores": dim_scores,
            "completion": completion,
            "saved_at": datetime.now().isoformat(),
        }
        
        # Save
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved high-score candidate: {filename} (reward={total_reward:.3f})")
    
    def get_saved_count(self) -> int:
        return self._saved_count


# Global high score saver instance
high_score_saver: Optional[HighScoreSaver] = None


class BatchStatsTracker:
    """
    Track statistics for each batch.
    """
    _instance = None
    HIGH_SCORE_THRESHOLD = 0.8
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._completions = {}  # idx -> info dict
        return cls._instance
    
    def start_batch(self):
        """Clear for new batch."""
        self._completions = {}
    
    def record(self, score: RewardScore, sample_info: Dict, completion: str, idx: int = None):
        """Record a candidate's score and info."""
        if idx is None:
            idx = len(self._completions)
        
        # Extract dimension scores
        dim_scores = {}
        if score.format_score:
            dim_scores["format"] = score.format_score.raw_score
        if score.summary_score:
            dim_scores["summary"] = score.summary_score.raw_score
        if score.action_score:
            dim_scores["action"] = score.action_score.raw_score
        if score.context_instruction_score:
            dim_scores["context_instruction"] = score.context_instruction_score.raw_score
        if score.context_namespace_score:
            dim_scores["context_namespace"] = score.context_namespace_score.raw_score
        if score.confidence_score:
            dim_scores["confidence"] = score.confidence_score.raw_score
        
        self._completions[idx] = {
            "total_reward": score.total_score,
            "dim_scores": dim_scores,
            "completion": completion,
            "sample_info": sample_info,
            "score_obj": score,
        }
    
    def log_batch_summary(self):
        """Log batch summary."""
        if not self._completions:
            return
        
        # Dimension abbreviations
        dim_abbrev = {
            "format": "fmt",
            "summary": "sum",
            "action": "act",
            "context_instruction": "ctx_i",
            "context_namespace": "ctx_n",
            "confidence": "conf",
        }
        
        # Group completions by sample (file + iter)
        samples_map = {}  # (problem_id, iter) -> list of (idx, info)
        for idx, info in self._completions.items():
            sample_info = info.get("sample_info", {})
            problem_id = sample_info.get("problem_id", "unknown")
            current_iter = sample_info.get("current_iter", 0)
            key = (problem_id, current_iter)
            if key not in samples_map:
                samples_map[key] = []
            samples_map[key].append((idx, info))
        
        # Log each sample's candidates
        for (problem_id, current_iter), completions_list in sorted(samples_map.items()):
            if not completions_list:
                continue
            
            # Get sample info from first completion
            _, first_info = completions_list[0]
            sample_info = first_info.get("sample_info", {})
            total_iters = sample_info.get("total_iters", "?")
            task_type = sample_info.get("task_type", "?")
            expected_action = sample_info.get("expected_action", "?")
            
            # Extract filename from file_path
            file_path = sample_info.get("file_path", "")
            filename = Path(file_path).stem if file_path else problem_id
            
            high_score_count = sum(
                1 for _, info in completions_list
                if info["total_reward"] >= self.HIGH_SCORE_THRESHOLD
            )
            
            logger.info(f"[Sample] file={filename} | iter={current_iter}/{total_iters} | type={task_type} | expected={expected_action} | Candidates ({high_score_count}/{len(completions_list)} >= {self.HIGH_SCORE_THRESHOLD}):")
            
            for local_idx, (global_idx, info) in enumerate(completions_list):
                total = info["total_reward"]
                dim_scores = info["dim_scores"]
                
                # Format dimension (0/1 -> check/cross)
                fmt_val = dim_scores.get("format", 0.0)
                fmt_str = "OK" if fmt_val >= 5.0 else "FAIL"
                
                # Build dimension string
                dim_parts = [f"fmt={fmt_str}"]
                for dim_name in ["summary", "action", "context_instruction", "context_namespace", "confidence"]:
                    abbr = dim_abbrev.get(dim_name, dim_name[:3])
                    score = dim_scores.get(dim_name, 0.0)
                    dim_parts.append(f"{abbr}={score:.1f}")
                
                dim_str = " ".join(dim_parts)
                marker = "*" if total >= self.HIGH_SCORE_THRESHOLD else " "
                logger.info(f"  {marker}#{local_idx+1}: {dim_str} | total={total:.3f}")
    
    def get_total_scores(self) -> List[float]:
        """Get all total scores."""
        return [info["total_reward"] for info in self._completions.values()]
    
    def get_high_score_ratio(self) -> float:
        """Get ratio of high-scoring candidates."""
        if not self._completions:
            return 0.0
        high_count = sum(1 for info in self._completions.values() if info["total_reward"] >= self.HIGH_SCORE_THRESHOLD)
        return high_count / len(self._completions)
    
    def get_dim_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each dimension."""
        stats = {}
        dim_names = ["format", "summary", "action", "context_instruction", "context_namespace", "confidence"]
        
        for dim_name in dim_names:
            scores = [info["dim_scores"].get(dim_name, 0.0) for info in self._completions.values() if dim_name in info["dim_scores"]]
            if scores:
                stats[dim_name] = {
                    "mean": sum(scores) / len(scores),
                    "max": max(scores),
                    "min": min(scores),
                }
        return stats
    
    def get_high_score_candidates(self) -> List[Dict]:
        """Get candidates with score >= threshold for saving."""
        candidates = []
        for idx, info in self._completions.items():
            if info["total_reward"] >= self.HIGH_SCORE_THRESHOLD:
                candidates.append({
                    "completion": info.get("completion", ""),
                    "sample_info": info.get("sample_info", {}),
                    "total_reward": info["total_reward"],
                    "dim_scores": info["dim_scores"],
                })
        return candidates


batch_stats = BatchStatsTracker()


class ObserverRewardCallback(TrainerCallback):
    """Callback for logging reward statistics to TensorBoard."""
    
    def __init__(self):
        self._tb_writer = None
    
    def on_step_begin(self, args, state, control, **kwargs):
        batch_stats.start_batch()
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        global high_score_saver
        log_config.increment_step()
        
        # Always log batch summary (every step)
        batch_stats.log_batch_summary()
        
        # Save high-scoring candidates
        if high_score_saver:
            for candidate in batch_stats.get_high_score_candidates():
                high_score_saver.save_candidate(
                    completion=candidate["completion"],
                    sample_info=candidate["sample_info"],
                    total_reward=candidate["total_reward"],
                    dim_scores=candidate["dim_scores"],
                )
        
        # Log to TensorBoard
        if self._tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                if hasattr(args, 'logging_dir') and args.logging_dir:
                    self._tb_writer = SummaryWriter(log_dir=args.logging_dir)
            except Exception as e:
                logger.warning(f"Could not create TensorBoard writer: {e}")
        
        total_scores = batch_stats.get_total_scores()
        if self._tb_writer and total_scores:
            # Total reward statistics
            self._tb_writer.add_scalar("reward/total_mean", sum(total_scores) / len(total_scores), state.global_step)
            self._tb_writer.add_scalar("reward/total_max", max(total_scores), state.global_step)
            self._tb_writer.add_scalar("reward/total_min", min(total_scores), state.global_step)
            
            # High score ratio
            self._tb_writer.add_scalar("reward/high_score_ratio", batch_stats.get_high_score_ratio(), state.global_step)
            
            # Per-dimension statistics
            dim_stats = batch_stats.get_dim_stats()
            for dim_name, stats in dim_stats.items():
                self._tb_writer.add_scalar(f"reward_dim/{dim_name}_mean", stats["mean"], state.global_step)
                self._tb_writer.add_scalar(f"reward_dim/{dim_name}_max", stats["max"], state.global_step)
            
            self._tb_writer.flush()
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        if self._tb_writer:
            self._tb_writer.close()
        return control


class MultiDimRewardWrapper:
    """
    Wrapper for multi-dimensional reward scoring.
    Called by TRL GRPOTrainer for each candidate.
    """
    
    __name__ = "observer_reward"
    
    def __init__(
        self,
        reward_cache: RewardScoreCache,
        sample_info_map: Dict[str, Dict],
    ):
        self.cache = reward_cache
        self.sample_info_map = sample_info_map
    
    def __call__(
        self,
        completions: List[str] = None,
        prompts: List[str] = None,
        **kwargs
    ) -> List[float]:
        """Score completions and return total rewards."""
        if completions is None:
            completions = kwargs.get("completions", [])
        if prompts is None:
            prompts = kwargs.get("prompts", [])
        
        rewards = []
        
        for idx, completion in enumerate(completions):
            if isinstance(completion, (list, tuple)):
                rewards.append(0.0)
                continue
            
            # Canonicalize completion to avoid repeated JSON blocks/noisy markdown
            completion_str = canonicalize_completion(completion)
            prompt_str = str(prompts[idx]) if idx < len(prompts) else ""
            
            # Get sample info from prompt
            sample_info = self.sample_info_map.get(hash(prompt_str), {})
            
            # Get score
            score = self.cache.get_score(completion_str, sample_info)
            
            # Record for batch stats (with idx)
            batch_stats.record(score, sample_info, completion_str, idx=idx)
            
            rewards.append(score.total_score)
        
        return rewards


def create_hf_dataset(
    data_loader: ObserverDataLoader,
    num_samples: Optional[int] = None,
) -> Dataset:
    """Create HuggingFace Dataset for TRL."""
    samples = create_dataset_from_loader(data_loader, num_samples)
    
    # Convert to HF Dataset
    data = []
    for sample in samples:
        data.append({
            "prompt": sample["prompt"],
            "_sample_info": json.dumps({
                "problem_id": sample["problem_id"],
                "task_type": sample["task_type"],
                "current_iter": sample["current_iter"],
                "total_iters": sample["total_iters"],
                "expected_action": sample["expected_action"],
                "execution_history": sample["execution_history"],
                "file_path": sample["file_path"],
            }),
        })
    
    return Dataset.from_list(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Observer GRPO Training")
    
    # Model
    parser.add_argument("--policy-model-path", type=str, default="Qwen/Qwen3-14B")
    
    # Reward
    parser.add_argument("--reward-model", type=str, default="anthropic/claude-sonnet-4.5")
    parser.add_argument("--reward-api-key", type=str, default=None,
                       help="Reward model API key (default: from OPENROUTER_API_KEY env var)")
    parser.add_argument("--proxy-url", type=str, default="",
                       help="HTTP proxy URL (leave empty if not needed)")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="./data/gt/gt_observer_training_best")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of training samples (None = use all data with balanced sampling)")
    
    # Training
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--gen-temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    
    # LoRA
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    
    # vLLM (always enabled)
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.5, help="vLLM GPU memory utilization")
    parser.add_argument("--vllm-max-model-len", type=int, default=14000, help="Max model length for vLLM")
    
    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/observer")
    parser.add_argument("--log-dir", type=str, default="./logs/observer_grpo")
    parser.add_argument("--high-score-dir", type=str, default="./data/gt/observer_training_high_score",
                       help="Directory to save high-scoring (>=0.8) candidates")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=10)
    
    # Other
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not TRL_AVAILABLE:
        print("Error: TRL not installed. Run: pip install trl>=0.12.0")
        return
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("""
+===============================================================+
|           Observer GRPO Training with TRL + vLLM              |
+===============================================================+
    """)
    
    # Set logging steps
    log_config.logging_steps = args.logging_steps
    
    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = ObserverGRPOConfig(
        policy_model_path=args.policy_model_path,
        reward_model_name=args.reward_model,
        reward_model_api_key=args.reward_api_key or os.getenv("OPENROUTER_API_KEY", ""),
        proxy_url=args.proxy_url,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        high_score_dir=args.high_score_dir,
        group_size=args.group_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        generation_temperature=args.gen_temperature,
        max_new_tokens=args.max_new_tokens,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_vllm=True,  # Always use vLLM
        vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        vllm_max_model_len=args.vllm_max_model_len,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    
    # Initialize high score saver
    global high_score_saver
    high_score_saver = HighScoreSaver(
        output_dir=config.high_score_dir,
        threshold=config.high_score_threshold,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loader
    logger.info(f"Loading data from {config.data_dir}")
    data_loader = ObserverDataLoader(
        data_dir=config.data_dir,
        prompts_path=config.prompts_path,
        batch_size=config.batch_size,
        group_size=config.group_size,
    )
    
    logger.info(f"Dataset stats: {data_loader.get_stats()}")
    
    # Create HuggingFace dataset
    dataset = create_hf_dataset(data_loader, args.num_samples)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Build sample info map (prompt hash -> sample info)
    # Also need to load full data for steps
    sample_info_map = {}
    for item in dataset:
        prompt_hash = hash(item["prompt"])
        info = json.loads(item["_sample_info"])
        
        # Load steps from original file
        file_path = info.get("file_path", "")
        if file_path and Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                info["steps"] = data.get("steps", [])
                info["task_description"] = data.get("task_description", "")
        else:
            info["steps"] = []
            info["task_description"] = ""
        
        sample_info_map[prompt_hash] = info
    
    # Create reward model
    reward_model = ObserverRewardModel(config)
    reward_cache = RewardScoreCache(reward_model)
    
    # Create reward wrapper
    reward_fn = MultiDimRewardWrapper(reward_cache, sample_info_map)
    
    # LoRA config
    peft_config = None
    if config.use_lora and PEFT_AVAILABLE:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        logger.info(f"Using LoRA with rank={config.lora_rank}")
    
    # TensorBoard log directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tensorboard_log_dir = f"{config.log_dir}/{run_timestamp}"
    
    # GRPOConfig - vLLM only mode (no accelerate/distributed support)
    import inspect
    grpo_sig = inspect.signature(GRPOConfig.__init__)
    grpo_params = list(grpo_sig.parameters.keys())
    
    # Model init kwargs for vLLM mode
    model_init_kwargs = {
        "torch_dtype": "bfloat16" if config.bf16 else "float16",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    
    grpo_config_kwargs = {
        "output_dir": config.checkpoint_dir,
        "model_init_kwargs": model_init_kwargs,
        "num_generations": config.group_size,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "bf16": config.bf16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "report_to": ["tensorboard"],
        "logging_dir": tensorboard_log_dir,
        "logging_first_step": True,
        "use_vllm": True,  # Always use vLLM
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
    }
    
    # Add vLLM params if available
    if config.use_vllm:
        if "vllm_mode" in grpo_params:
            grpo_config_kwargs["vllm_mode"] = "colocate"
        if "vllm_gpu_memory_utilization" in grpo_params:
            grpo_config_kwargs["vllm_gpu_memory_utilization"] = config.vllm_gpu_memory_utilization
        if "vllm_tensor_parallel_size" in grpo_params:
            grpo_config_kwargs["vllm_tensor_parallel_size"] = config.vllm_tensor_parallel_size
        if "vllm_max_model_length" in grpo_params:
            grpo_config_kwargs["vllm_max_model_length"] = config.vllm_max_model_len
        if "vllm_enforce_eager" in grpo_params:
            grpo_config_kwargs["vllm_enforce_eager"] = True
    
    # Add max completion length
    if "max_completion_length" in grpo_params:
        grpo_config_kwargs["max_completion_length"] = config.max_new_tokens
    elif "response_length" in grpo_params:
        grpo_config_kwargs["response_length"] = config.max_new_tokens
    
    grpo_config = GRPOConfig(**grpo_config_kwargs)
    
    logger.info(f"Configuration:")
    logger.info(f"  Policy Model: {config.policy_model_path}")
    logger.info(f"  Reward Model: {config.reward_model_name}")
    logger.info(f"  Group Size: {config.group_size}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  vLLM Max Model Len: {config.vllm_max_model_len}")
    logger.info(f"  Use LoRA: {config.use_lora}")
    logger.info(f"  High Score Dir: {config.high_score_dir}")
    logger.info(f"  TensorBoard: {tensorboard_log_dir}")
    
    # Create trainer (vLLM mode - pass model path, TRL handles loading)
    trainer = GRPOTrainer(
        model=config.policy_model_path,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    # Add callback
    trainer.add_callback(ObserverRewardCallback())
    
    logger.info("Starting training...")
    
    # Train
    checkpoint_path = args.resume_from_checkpoint
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_path)
    
    try:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM: {e}")
        logger.error("Try reducing --batch-size, --group-size, or --max-new-tokens")
        raise
    
    # Save
    trainer.save_model(config.checkpoint_dir)
    logger.info(f"Model saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
