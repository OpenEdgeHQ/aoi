# GRPO Trainer for AOI Evolver
"""
Group Relative Policy Optimization (GRPO) Trainer.

GRPO is a reinforcement learning algorithm that:
1. Generates multiple candidates (a "group") per prompt
2. Uses an external reward model to score each candidate
3. Computes relative advantages within each group
4. Updates the policy using these advantages

Key advantages over PPO:
- No need for a separate critic/value network
- Naturally maintains generation diversity
- Lower memory footprint
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not available. Install with: pip install tensorboard")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available. LoRA fine-tuning will be disabled.")

from .grpo_config import GRPOConfig, RewardScoreConfig, DEFAULT_GRPO_CONFIG
from .data_loader import SeedScenario, GRPODataLoader, create_grpo_prompt
from .reward_model import RewardModel, ScenarioScore

logger = logging.getLogger(__name__)


@dataclass
class GRPOBatchOutput:
    """Output from a single GRPO training step."""
    loss: float
    policy_loss: float
    kl_loss: float
    entropy_loss: float
    mean_reward: float
    mean_advantage: float
    num_candidates: int
    accepted_candidates: int


@dataclass
class GeneratedCandidate:
    """A generated candidate with its metadata."""
    seed_idx: int
    candidate_idx: int
    prompt: str
    response: str
    log_prob: float
    reward: Optional[float] = None
    advantage: Optional[float] = None
    parsed_scenario: Optional[Dict[str, Any]] = None


class GRPOTrainer:
    """
    GRPO Trainer for fine-tuning LLMs on scenario generation.
    
    The training loop:
    1. For each seed in batch:
       a. Generate N candidates using the policy model
       b. Score each candidate using the reward model (SOTA LLM)
       c. Compute group-relative advantages
    2. Compute policy gradient loss weighted by advantages
    3. Update policy model
    """
    
    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        reward_config: Optional[RewardScoreConfig] = None,
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            config: GRPO training configuration
            reward_config: Reward scoring configuration
        """
        self.config = config or DEFAULT_GRPO_CONFIG
        self.reward_config = reward_config or RewardScoreConfig()
        
        # Set random seeds
        self._set_seed(self.config.seed)
        
        # Initialize components (lazy loading)
        self._policy_model = None
        self._ref_model = None
        self._tokenizer = None
        self._reward_model = None
        self._optimizer = None
        self._scheduler = None
        self._data_loader = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float('-inf')
        
        # Fault dimensions for diversity
        self.fault_dimensions = [
            "network_delay", "network_loss", "disk_io", "memory_pressure",
            "cpu_throttle", "pod_failure", "service_misconfiguration",
            "authentication_issue", "database_connection", "resource_exhaustion",
        ]
        
        # Logging
        self.train_history = []
        
        # TensorBoard writer
        self._tb_writer = None
        if TENSORBOARD_AVAILABLE:
            tb_log_dir = Path(self.config.log_dir) / "tensorboard" / datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            logger.info(f"TensorBoard logging enabled: {tb_log_dir}")
            logger.info(f"Run 'tensorboard --logdir={self.config.log_dir}/tensorboard' to view")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_policy_model(self):
        """Load the policy model."""
        logger.info(f"Loading policy model: {self.config.policy_model_name}")
        
        model_path = self.config.policy_model_path or self.config.policy_model_name
        
        # Quantization config for memory efficiency
        # Note: Disabled for newer GPUs (sm_120+) that bitsandbytes doesn't support yet
        quantization_config = None
        use_quantization = False
        
        if self.config.use_lora and use_quantization:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                logger.warning(f"Failed to create quantization config: {e}")
                quantization_config = None
        
        # Determine dtype and device_map
        if self.config.device == "cpu":
            dtype = torch.float32  # CPU doesn't support bf16 efficiently
            device_map = None
        else:
            dtype = torch.bfloat16 if self.config.bf16 else torch.float16
            device_map = "auto"
        
        # Load model
        self._policy_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Move to device if CPU
        if self.config.device == "cpu":
            self._policy_model = self._policy_model.to("cpu")
        
        # Apply LoRA if configured
        if self.config.use_lora:
            if PEFT_AVAILABLE:
                logger.info("=" * 60)
                logger.info("Applying LoRA adapter...")
                logger.info(f"  LoRA Rank: {self.config.lora_rank}")
                logger.info(f"  LoRA Alpha: {self.config.lora_alpha}")
                logger.info(f"  LoRA Dropout: {self.config.lora_dropout}")
                logger.info(f"  Target Modules: {self.config.lora_target_modules}")
                logger.info("=" * 60)
                
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules,
                    bias="none",
                )
                self._policy_model = get_peft_model(self._policy_model, lora_config)
                self._policy_model.print_trainable_parameters()
                logger.info("LoRA adapter applied successfully!")
                
                # Enable gradient checkpointing to reduce memory usage
                if hasattr(self._policy_model, 'enable_input_require_grads'):
                    self._policy_model.enable_input_require_grads()
                if hasattr(self._policy_model, 'gradient_checkpointing_enable'):
                    self._policy_model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled for memory efficiency")
            else:
                logger.error("=" * 60)
                logger.error("PEFT NOT AVAILABLE - LoRA DISABLED!")
                logger.error("This will cause OOM with large models!")
                logger.error("Install PEFT: pip install peft>=0.7.0")
                logger.error("=" * 60)
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        logger.info("Policy model loaded successfully")
    
    def _load_reference_model(self):
        """Load the reference model for KL divergence computation."""
        if self.config.kl_coef <= 0:
            logger.info("=" * 60)
            logger.info("KL coefficient is 0, skipping reference model loading")
            logger.info("This saves ~50% GPU memory but disables KL regularization")
            logger.info("=" * 60)
            self._ref_model = None
            return
        
        logger.info("Loading reference model for KL computation...")
        logger.warning("NOTE: Reference model uses significant GPU memory.")
        logger.warning("Set --kl-coef 0 to disable and save ~50% memory.")
        
        model_path = self.config.policy_model_path or self.config.policy_model_name
        
        # Determine dtype and device_map
        if self.config.device == "cpu":
            dtype = torch.float32
            device_map = None
        else:
            dtype = torch.bfloat16 if self.config.bf16 else torch.float16
            device_map = "auto"
        
        # Load reference model (frozen)
        self._ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Move to device if CPU
        if self.config.device == "cpu":
            self._ref_model = self._ref_model.to("cpu")
        
        self._ref_model.eval()
        for param in self._ref_model.parameters():
            param.requires_grad = False
        
        logger.info("Reference model loaded successfully")
    
    def _init_reward_model(self):
        """Initialize the reward model."""
        from .evolver_config import EvolverConfig
        
        evolver_config = EvolverConfig(
            api_key=self.config.reward_model_api_key,
            api_base=self.config.reward_model_api_base,
            reward_model=self.config.reward_model_name,
            reward_temperature=self.config.reward_temperature,
        )
        
        self._reward_model = RewardModel(evolver_config, self.reward_config)
        logger.info(f"Reward model initialized: {self.config.reward_model_name}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        # Get trainable parameters
        if self.config.use_lora and PEFT_AVAILABLE:
            params = [p for p in self._policy_model.parameters() if p.requires_grad]
        else:
            params = self._policy_model.parameters()
        
        # Create optimizer
        self._optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Create scheduler
        total_steps = self._estimate_total_steps()
        
        if self.config.scheduler_type.value == "cosine":
            self._scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.1,
            )
        else:
            self._scheduler = LinearLR(
                self._optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps,
            )
        
        logger.info(f"Optimizer initialized: {self.config.optimizer_type.value}")
        logger.info(f"Scheduler initialized: {self.config.scheduler_type.value}")
    
    def _estimate_total_steps(self) -> int:
        """Estimate total training steps."""
        if self.config.max_steps:
            return self.config.max_steps
        
        if self._data_loader:
            steps_per_epoch = len(self._data_loader) // self.config.gradient_accumulation_steps
            return steps_per_epoch * self.config.num_epochs
        
        return 1000  # Default fallback
    
    def _init_data_loader(self):
        """Initialize the data loader."""
        self._data_loader = GRPODataLoader(
            seed_dir=self.config.seed_data_dir,
            batch_size=self.config.batch_size,
            group_size=self.config.group_size,
            max_seeds=self.config.max_seeds,
            shuffle=True,
            filter_successful=True,
        )
        
        stats = self._data_loader.get_stats()
        logger.info(f"Data loader stats: {stats}")
    
    def setup(self):
        """Set up all components for training."""
        logger.info("Setting up GRPO trainer...")
        
        # Initialize data loader first (needed for step estimation)
        self._init_data_loader()
        
        # Load models
        self._load_policy_model()
        self._load_reference_model()
        
        # Initialize reward model
        self._init_reward_model()
        
        # Initialize optimizer
        self._init_optimizer()
        
        logger.info("GRPO trainer setup complete")
    
    def _select_fault_dimensions(self, n: int) -> List[str]:
        """Select diverse fault dimensions for generation."""
        dims = self.fault_dimensions.copy()
        random.shuffle(dims)
        return dims[:n]
    
    @torch.no_grad()
    def _generate_candidates(
        self,
        seed: SeedScenario,
        num_candidates: int,
    ) -> List[GeneratedCandidate]:
        """
        Generate multiple candidates for a seed.
        
        Args:
            seed: The seed scenario
            num_candidates: Number of candidates to generate
            
        Returns:
            List of GeneratedCandidate objects
        """
        candidates = []
        fault_dims = self._select_fault_dimensions(num_candidates)
        
        # Determine device
        if self.config.device == "cpu":
            device = torch.device("cpu")
        else:
            device = self._policy_model.device
        
        # ============================================================
        # IMPORTANT: Disable gradient checkpointing during generation
        # Gradient checkpointing is incompatible with KV cache (use_cache=True)
        # which is needed for efficient text generation
        # ============================================================
        was_gradient_checkpointing = False
        if hasattr(self._policy_model, 'is_gradient_checkpointing') and self._policy_model.is_gradient_checkpointing:
            was_gradient_checkpointing = True
            self._policy_model.gradient_checkpointing_disable()
            logger.debug("Temporarily disabled gradient checkpointing for generation")
        
        # Set model to eval mode for generation
        self._policy_model.eval()
        
        for i, fault_dim in enumerate(fault_dims):
            prompt = create_grpo_prompt(seed, fault_dim)
            
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_new_tokens * 2,  # Leave room for response
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with use_cache=True for efficient generation
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,  # Enable KV cache for generation
            )
            
            outputs = self._policy_model.generate(
                **inputs,
                generation_config=generation_config,
            )
            
            # Extract response
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Compute log probability
            log_prob = self._compute_log_prob(inputs["input_ids"], outputs.sequences)
            
            # Parse response
            parsed = self._parse_response(response)
            
            candidates.append(GeneratedCandidate(
                seed_idx=0,  # Will be set by caller
                candidate_idx=i,
                prompt=prompt,
                response=response,
                log_prob=log_prob,
                parsed_scenario=parsed,
            ))
        
        # ============================================================
        # Restore gradient checkpointing and training mode after generation
        # ============================================================
        if was_gradient_checkpointing:
            self._policy_model.gradient_checkpointing_enable()
            logger.debug("Re-enabled gradient checkpointing after generation")
        
        # Restore training mode (will be set properly in train_step)
        self._policy_model.train()
        
        return candidates
    
    def _compute_log_prob(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> float:
        """Compute log probability of generated sequence."""
        # Forward pass to get logits
        with torch.no_grad():
            outputs = self._policy_model(output_ids)
            logits = outputs.logits
        
        # Compute log probs for generated tokens only
        gen_start = input_ids.shape[1]
        gen_logits = logits[:, gen_start-1:-1, :]  # Shift for next token prediction
        gen_ids = output_ids[:, gen_start:]
        
        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=gen_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probs (or mean for length normalization)
        total_log_prob = token_log_probs.sum().item()
        
        return total_log_prob
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response."""
        if not response:
            logger.warning("Empty response from policy model")
            return None
        
        # Log the response for debugging
        logger.info(f"Policy model response (first 300 chars): {response[:300]}...")
        
        try:
            json_text = self._extract_json(response)
            if json_text:
                # Try to fix truncated JSON
                json_text = self._fix_truncated_json(json_text)
                result = json.loads(json_text)
                logger.info(f"Successfully parsed JSON with keys: {list(result.keys())}")
                return result
            else:
                logger.warning(f"No JSON found in response")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            # Try more aggressive fixing
            json_text = self._extract_json(response)
            if json_text:
                fixed = self._aggressive_json_fix(json_text)
                if fixed:
                    try:
                        result = json.loads(fixed)
                        logger.info(f"Parsed JSON after aggressive fix with keys: {list(result.keys())}")
                        return result
                    except:
                        pass
            return None
    
    def _fix_truncated_json(self, json_str: str) -> str:
        """Fix common JSON truncation issues."""
        if not json_str:
            return json_str
        
        json_str = json_str.strip()
        
        # Remove trailing comma before closing brackets
        json_str = json_str.rstrip(',')
        
        # Count brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Add missing closing brackets/braces
        while close_brackets < open_brackets:
            json_str += ']'
            close_brackets += 1
        
        while close_braces < open_braces:
            json_str += '}'
            close_braces += 1
        
        return json_str
    
    def _aggressive_json_fix(self, json_str: str) -> Optional[str]:
        """Aggressively try to fix broken JSON."""
        if not json_str:
            return None
        
        import re
        
        # Fix common issues
        json_str = json_str.strip()
        
        # Remove trailing incomplete strings
        # If the string ends with an incomplete value like "some text
        if json_str.count('"') % 2 != 0:
            # Find the last complete string
            last_quote = json_str.rfind('"')
            if last_quote > 0:
                # Check if it's a closing quote
                before = json_str[last_quote-1] if last_quote > 0 else ''
                if before == '\\':
                    # This is an escaped quote, find the one before
                    json_str = json_str[:last_quote-1] + '"'
                else:
                    # Try to close at the last proper string boundary
                    # Look for the last colon followed by a quote
                    pattern = r':\s*"[^"]*$'
                    match = re.search(pattern, json_str)
                    if match:
                        json_str = json_str[:match.end()-1] + '"'
        
        # Remove trailing comma and whitespace
        json_str = re.sub(r',\s*$', '', json_str)
        
        # Close any unclosed arrays or objects
        json_str = self._fix_truncated_json(json_str)
        
        return json_str
    
    def _extract_json(self, response: str) -> Optional[str]:
        """Extract JSON from response text."""
        if not response:
            return None
        
        response = response.strip()
        
        # Try to find JSON in code blocks first (4 backticks have priority)
        if "````json" in response:
            json_start = response.find("````json") + 8
            json_end = response.find("````", json_start)
            if json_end > json_start:
                return response[json_start:json_end].strip()
            else:
                # No closing ````, take everything after ````json
                return response[json_start:].strip()
        
        if "````" in response:
            json_start = response.find("````") + 4
            json_end = response.find("````", json_start)
            if json_end > json_start:
                extracted = response[json_start:json_end].strip()
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted
            else:
                extracted = response[json_start:].strip()
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted
        
        # Try 3 backticks
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_end > json_start:
                return response[json_start:json_end].strip()
            else:
                # No closing ```, take everything after ```json
                return response[json_start:].strip()
        
        if "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            if json_end > json_start:
                extracted = response[json_start:json_end].strip()
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted
            else:
                # No closing ```, take everything after first ```
                extracted = response[json_start:].strip()
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted
        
        # Find JSON object directly in the response
        brace_start = response.find("{")
        if brace_start >= 0:
            # Take everything from the first brace
            return response[brace_start:]
        
        return None
    
    async def _score_candidates(
        self,
        candidates: List[GeneratedCandidate],
    ) -> List[ScenarioScore]:
        """Score candidates using the reward model."""
        # Prepare candidates for scoring - only include valid parsed scenarios
        to_score = []
        valid_indices = []
        
        for i, c in enumerate(candidates):
            if c.parsed_scenario:
                # Add required fields for scoring if missing
                scenario = c.parsed_scenario.copy()
                if "fault_dimension" not in scenario:
                    scenario["fault_dimension"] = "unknown"
                if "expected_findings" not in scenario:
                    scenario["expected_findings"] = scenario.get("task_info", {}).get("description", "N/A")
                if "root_cause" not in scenario:
                    scenario["root_cause"] = scenario.get("task_info", {}).get("description", "N/A")
                if "difficulty_level" not in scenario:
                    scenario["difficulty_level"] = "medium"
                if "candidate_id" not in scenario:
                    scenario["candidate_id"] = i
                
                to_score.append(scenario)
                valid_indices.append(i)
            else:
                logger.warning(f"Candidate {i} has no parsed_scenario, assigning default score")
        
        if not to_score:
            logger.warning("No valid scenarios to score!")
            # Return default scores for all candidates
            return [
                ScenarioScore(
                    scenario_id=i,
                    logical_consistency=3.0,  # Low score for failed parses
                    complexity_value=3.0,
                    syntax_correctness=0.0,
                    overall_score=2.0,
                    recommendation="reject",
                    reasoning={"error": "Failed to parse generated scenario"},
                    improvement_suggestions=[],
                )
                for i in range(len(candidates))
            ]
        
        # Score valid scenarios
        valid_scores = await self._reward_model.score_batch(to_score)
        
        # Map scores back to all candidates
        scores = []
        valid_idx = 0
        for i in range(len(candidates)):
            if i in valid_indices:
                scores.append(valid_scores[valid_idx])
                valid_idx += 1
            else:
                # Default low score for failed parses
                scores.append(ScenarioScore(
                    scenario_id=i,
                    logical_consistency=3.0,
                    complexity_value=3.0,
                    syntax_correctness=0.0,
                    overall_score=2.0,
                    recommendation="reject",
                    reasoning={"error": "Failed to parse generated scenario"},
                    improvement_suggestions=[],
                ))
        
        return scores
    
    def _compute_grpo_advantages(
        self,
        rewards: List[float],
    ) -> List[float]:
        """
        Compute GRPO advantages.
        
        GRPO advantage: A_i = (R_i - mean(R)) / std(R)
        
        Args:
            rewards: List of reward values for the group
            
        Returns:
            List of advantage values
        """
        if len(rewards) == 0:
            return []
        
        rewards_np = np.array(rewards)
        mean_reward = np.mean(rewards_np)
        std_reward = np.std(rewards_np) + 1e-8  # Prevent division by zero
        
        advantages = (rewards_np - mean_reward) / std_reward
        
        return advantages.tolist()
    
    def _compute_multidim_advantages(
        self,
        scores: List['ScenarioScore'],
    ) -> List[float]:
        """
        Compute GRPO advantages from multi-dimensional scores.
        
        Each dimension gets its own advantage, then we combine them with weights.
        This allows the model to learn from diverse signals.
        
        NEW WEIGHTS (aligned with training weights, excluding diversity which is constant here):
        - solution_effectiveness: 0.375  (0.30 / 0.80)
        - commands_completeness: 0.250  (0.20 / 0.80)
        - problem_validity: 0.125      (0.10 / 0.80)
        - commands_correctness: 0.125  (0.10 / 0.80)
        - format: 0.125                (0.10 / 0.80)
        
        Args:
            scores: List of ScenarioScore objects with multi-dimensional scores
            
        Returns:
            List of combined advantage values
        """
        if len(scores) == 0:
            return []
        
        # Dimension weights (aligned with current training weights; renormalized)
        weights = {
            "solution_effectiveness": 0.375,
            "commands_completeness": 0.250,
            "problem_validity": 0.125,
            "commands_correctness": 0.125,
            "format": 0.125,
        }
        
        # Extract scores by NEW dimensions
        problem_validity_scores = np.array([s.problem_validity_score for s in scores])
        commands_completeness_scores = np.array([s.commands_completeness_score for s in scores])
        commands_correctness_scores = np.array([s.commands_correctness_score for s in scores])
        solution_effectiveness_scores = np.array([s.solution_effectiveness_score for s in scores])
        format_scores = np.array([s.format_score for s in scores])
        
        # Compute advantage for each dimension
        def compute_adv(scores_arr):
            mean = np.mean(scores_arr)
            std = np.std(scores_arr) + 1e-8
            return (scores_arr - mean) / std
        
        problem_validity_adv = compute_adv(problem_validity_scores)
        commands_completeness_adv = compute_adv(commands_completeness_scores)
        commands_correctness_adv = compute_adv(commands_correctness_scores)
        solution_effectiveness_adv = compute_adv(solution_effectiveness_scores)
        format_adv = compute_adv(format_scores)
        
        # Combine with NEW weights
        combined_advantages = (
            solution_effectiveness_adv * weights["solution_effectiveness"] +
            commands_completeness_adv * weights["commands_completeness"] +
            problem_validity_adv * weights["problem_validity"] +
            commands_correctness_adv * weights["commands_correctness"] +
            format_adv * weights["format"]
        )
        
        # Log dimension-wise stats
        logger.debug(
            f"Dimension advantages - "
            f"cmds_complete: {commands_completeness_adv.mean():.2f}, "
            f"solution: {solution_effectiveness_adv.mean():.2f}, "
            f"problem: {problem_validity_adv.mean():.2f}, "
            f"cmds_correct: {commands_correctness_adv.mean():.2f}, "
            f"format: {format_adv.mean():.2f}"
        )
        
        return combined_advantages.tolist()
    
    def _compute_kl_divergence(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference."""
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        kl = torch.sum(
            torch.exp(policy_log_probs) * (policy_log_probs - ref_log_probs),
            dim=-1
        )
        return kl.mean()
    
    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the distribution."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean()
    
    async def train_step(
        self,
        seeds: List[SeedScenario],
    ) -> GRPOBatchOutput:
        """
        Perform a single GRPO training step.
        
        Args:
            seeds: Batch of seed scenarios
            
        Returns:
            GRPOBatchOutput with loss and metrics
        """
        self._policy_model.train()
        
        all_candidates = []
        all_rewards = []
        all_advantages = []
        
        # Generate and score candidates for each seed
        for seed_idx, seed in enumerate(seeds):
            # Generate candidates
            candidates = self._generate_candidates(seed, self.config.group_size)
            for c in candidates:
                c.seed_idx = seed_idx
            
            # Score candidates
            scores = await self._score_candidates(candidates)
            
            # Extract rewards
            rewards = [s.overall_score for s in scores]
            
            # Compute multi-dimensional advantages
            # This uses per-dimension GRPO to get more nuanced learning signals
            advantages = self._compute_multidim_advantages(scores)
            
            # Log NEW dimension scores for monitoring
            logger.info(f"\n{'─'*70}")
            logger.info(f"Seed {seed_idx} ({seed.problem_id}) - Candidate Scores:")
            logger.info(f"{'─'*70}")
            for i, (s, adv) in enumerate(zip(scores, advantages)):
                logger.info(
                    f"  Candidate {i}: "
                    f"problem={s.problem_validity_score:.1f}, "
                    f"cmds_complete={s.commands_completeness_score:.1f}, "
                    f"cmds_correct={s.commands_correctness_score:.1f}, "
                    f"solution={s.solution_effectiveness_score:.1f}, "
                    f"format={s.format_score:.1f} | "
                    f"overall={s.overall_score:.2f}, adv={adv:+.3f}"
                )
            logger.info(f"  Group Mean Reward: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
            logger.info(f"{'─'*70}\n")
            
            # TensorBoard logging for candidate scores
            if self._tb_writer is not None:
                batch_step = self.global_step * len(seeds) + seed_idx
                for i, s in enumerate(scores):
                    self._tb_writer.add_scalars(
                        f"Candidates/seed_{seed_idx}/candidate_{i}",
                        {
                            "problem_validity": s.problem_validity_score,
                            "commands_completeness": s.commands_completeness_score,
                            "commands_correctness": s.commands_correctness_score,
                            "solution_effectiveness": s.solution_effectiveness_score,
                            "format": s.format_score,
                            "overall": s.overall_score,
                        },
                        batch_step
                    )
                # Log group statistics
                self._tb_writer.add_scalar(f"Group/seed_{seed_idx}/mean_reward", np.mean(rewards), batch_step)
                self._tb_writer.add_scalar(f"Group/seed_{seed_idx}/std_reward", np.std(rewards), batch_step)
            
            # Update candidates
            for c, r, a in zip(candidates, rewards, advantages):
                c.reward = r
                c.advantage = a
                all_candidates.append(c)
                all_rewards.append(r)
                all_advantages.append(a)
        
        # Determine device
        if self.config.device == "cpu":
            device = torch.device("cpu")
        else:
            device = self._policy_model.device
        
        # ============================================================
        # MEMORY-EFFICIENT GRADIENT ACCUMULATION
        # Process candidates one at a time and accumulate gradients
        # This prevents OOM by not keeping all computation graphs in memory
        # ============================================================
        
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_entropy_loss = 0.0
        num_valid = 0
        
        # Count valid candidates first for proper loss scaling
        valid_candidates = [c for c in all_candidates 
                          if c.parsed_scenario is not None and c.advantage is not None]
        
        if len(valid_candidates) == 0:
            logger.warning("No valid candidates for loss computation!")
            return GRPOBatchOutput(
                loss=0.0,
                policy_loss=0.0,
                kl_loss=0.0,
                entropy_loss=0.0,
                mean_reward=np.mean(all_rewards) if all_rewards else 0.0,
                mean_advantage=np.mean(all_advantages) if all_advantages else 0.0,
                num_candidates=len(all_candidates),
                accepted_candidates=0,
            )
        
        # Scale factor for gradient accumulation across candidates
        scale_factor = 1.0 / len(valid_candidates)
        
        for candidate_idx, candidate in enumerate(all_candidates):
            if candidate.parsed_scenario is None:
                logger.warning(f"Skipping candidate {candidate.candidate_idx} - failed to parse")
                continue
            
            if candidate.advantage is None:
                logger.warning(f"Skipping candidate {candidate.candidate_idx} - no advantage")
                continue
            
            # Tokenize prompt + response
            full_text = candidate.prompt + candidate.response
            inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_new_tokens * 2,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass (with gradients)
            outputs = self._policy_model(**inputs)
            logits = outputs.logits
            
            # Compute log prob of response
            prompt_len = len(self._tokenizer.encode(candidate.prompt))
            
            # Handle edge cases where response might be empty
            if prompt_len >= inputs["input_ids"].shape[1]:
                logger.warning(f"Skipping candidate {candidate.candidate_idx} - response too short")
                # Clean up tensors
                del outputs, logits, inputs
                continue
            
            response_logits = logits[:, prompt_len-1:-1, :]
            response_ids = inputs["input_ids"][:, prompt_len:]
            
            # Make sure we have something to compute
            if response_ids.shape[1] == 0 or response_logits.shape[1] == 0:
                logger.warning(f"Skipping candidate {candidate.candidate_idx} - empty response tokens")
                # Clean up tensors
                del outputs, logits, inputs, response_logits, response_ids
                continue
            
            # Align dimensions
            min_len = min(response_logits.shape[1], response_ids.shape[1])
            response_logits = response_logits[:, :min_len, :]
            response_ids = response_ids[:, :min_len]
            
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Policy gradient loss: -advantage * log_prob
            pg_loss = -candidate.advantage * token_log_probs.mean()
            
            # KL divergence loss
            kl_loss = torch.tensor(0.0, device=device)
            if self._ref_model is not None and self.config.kl_coef > 0:
                with torch.no_grad():
                    ref_outputs = self._ref_model(**inputs)
                    ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
                    ref_logits = ref_logits[:, :min_len, :]
                
                kl_loss = self.config.kl_coef * self._compute_kl_divergence(response_logits, ref_logits)
                
                # Clean up reference model outputs
                del ref_outputs, ref_logits
            
            # Entropy bonus
            entropy_loss = torch.tensor(0.0, device=device)
            if self.config.entropy_coef > 0:
                entropy = self._compute_entropy(response_logits)
                entropy_loss = -self.config.entropy_coef * entropy
            
            # Compute scaled loss for this candidate
            candidate_loss = (pg_loss + kl_loss + entropy_loss) * scale_factor
            
            # Backward pass for this candidate - accumulate gradients
            if candidate_loss.requires_grad:
                candidate_loss.backward()
            
            # Track losses (detached values for logging)
            total_policy_loss += pg_loss.item() * scale_factor
            total_kl_loss += kl_loss.item() * scale_factor
            total_entropy_loss += entropy_loss.item() * scale_factor
            num_valid += 1
            
            # ============================================================
            # CRITICAL: Clean up tensors to free GPU memory
            # ============================================================
            del outputs, logits, inputs
            del response_logits, response_ids, log_probs, token_log_probs
            del pg_loss, kl_loss, entropy_loss, candidate_loss
            
            # Periodically clear CUDA cache to prevent fragmentation
            if candidate_idx % 4 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Gradient clipping (after all gradients are accumulated)
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._policy_model.parameters(),
                self.config.max_grad_norm
            )
        
        # Final cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        total_loss = total_policy_loss + total_kl_loss + total_entropy_loss
        
        return GRPOBatchOutput(
            loss=total_loss,
            policy_loss=total_policy_loss,
            kl_loss=total_kl_loss,
            entropy_loss=total_entropy_loss,
            mean_reward=np.mean(all_rewards) if all_rewards else 0.0,
            mean_advantage=np.mean(all_advantages) if all_advantages else 0.0,
            num_candidates=len(all_candidates),
            accepted_candidates=num_valid,
        )
    
    async def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {
            "loss": [],
            "policy_loss": [],
            "kl_loss": [],
            "entropy_loss": [],
            "mean_reward": [],
            "acceptance_rate": [],
        }
        
        self._optimizer.zero_grad()
        accumulated_steps = 0
        total_batches = len(self._data_loader)
        
        for batch_idx, seeds in enumerate(self._data_loader):
            # Train step
            output = await self.train_step(seeds)
            
            # Calculate acceptance rate
            acceptance_rate = output.accepted_candidates / output.num_candidates if output.num_candidates > 0 else 0
            
            # Accumulate metrics
            epoch_metrics["loss"].append(output.loss)
            epoch_metrics["policy_loss"].append(output.policy_loss)
            epoch_metrics["kl_loss"].append(output.kl_loss)
            epoch_metrics["entropy_loss"].append(output.entropy_loss)
            epoch_metrics["mean_reward"].append(output.mean_reward)
            epoch_metrics["acceptance_rate"].append(acceptance_rate)
            
            # ============================================================
            # Log EVERY batch to TensorBoard for detailed monitoring
            # ============================================================
            if self._tb_writer is not None:
                batch_global_idx = self.epoch * total_batches + batch_idx
                self._tb_writer.add_scalar("Batch/loss", output.loss, batch_global_idx)
                self._tb_writer.add_scalar("Batch/policy_loss", output.policy_loss, batch_global_idx)
                self._tb_writer.add_scalar("Batch/kl_loss", output.kl_loss, batch_global_idx)
                self._tb_writer.add_scalar("Batch/entropy_loss", output.entropy_loss, batch_global_idx)
                self._tb_writer.add_scalar("Batch/mean_reward", output.mean_reward, batch_global_idx)
                self._tb_writer.add_scalar("Batch/acceptance_rate", acceptance_rate, batch_global_idx)
            
            # Quick console log for each batch
            logger.info(
                f"[Epoch {self.epoch+1}][Batch {batch_idx+1}/{total_batches}] "
                f"loss={output.loss:.4f}, reward={output.mean_reward:.2f}, accept={acceptance_rate:.0%}"
            )
            
            accumulated_steps += 1
            
            # Gradient accumulation
            if accumulated_steps >= self.config.gradient_accumulation_steps:
                self._optimizer.step()
                self._scheduler.step()
                self._optimizer.zero_grad()
                accumulated_steps = 0
                self.global_step += 1
                
                # Detailed logging at each optimization step
                if self.global_step % self.config.logging_steps == 0:
                    self._log_step(epoch_metrics)
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                # Check max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break
        
        # Handle remaining accumulated gradients
        if accumulated_steps > 0:
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()
            self.global_step += 1
            self._log_step(epoch_metrics)
        
        # Aggregate epoch metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def _log_step(self, metrics: Dict[str, List[float]]):
        """Log training step with detailed loss and reward information."""
        avg_metrics = {k: np.mean(v[-self.config.logging_steps:]) for k, v in metrics.items()}
        
        # Detailed console logging
        logger.info("=" * 70)
        logger.info(f"[Step {self.global_step}] Training Metrics:")
        logger.info("-" * 70)
        logger.info(f"  Total Loss:      {avg_metrics['loss']:.6f}")
        logger.info(f"  Policy Loss:     {avg_metrics['policy_loss']:.6f}")
        logger.info(f"  KL Loss:         {avg_metrics['kl_loss']:.6f}")
        logger.info(f"  Entropy Loss:    {avg_metrics['entropy_loss']:.6f}")
        logger.info("-" * 70)
        logger.info(f"  Mean Reward:     {avg_metrics['mean_reward']:.4f}")
        logger.info(f"  Acceptance Rate: {avg_metrics['acceptance_rate']:.2%}")
        logger.info(f"  Learning Rate:   {self._scheduler.get_last_lr()[0]:.2e}")
        logger.info("=" * 70)
        
        # TensorBoard logging
        if self._tb_writer is not None:
            # Loss metrics
            self._tb_writer.add_scalar("Loss/total", avg_metrics['loss'], self.global_step)
            self._tb_writer.add_scalar("Loss/policy", avg_metrics['policy_loss'], self.global_step)
            self._tb_writer.add_scalar("Loss/kl", avg_metrics['kl_loss'], self.global_step)
            self._tb_writer.add_scalar("Loss/entropy", avg_metrics['entropy_loss'], self.global_step)
            
            # Reward metrics
            self._tb_writer.add_scalar("Reward/mean", avg_metrics['mean_reward'], self.global_step)
            self._tb_writer.add_scalar("Reward/acceptance_rate", avg_metrics['acceptance_rate'], self.global_step)
            
            # Learning rate
            self._tb_writer.add_scalar("Train/learning_rate", self._scheduler.get_last_lr()[0], self.global_step)
            
            self._tb_writer.flush()
        
        self.train_history.append({
            "step": self.global_step,
            **avg_metrics
        })
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint-{self.global_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.config.use_lora and PEFT_AVAILABLE:
            self._policy_model.save_pretrained(checkpoint_path)
        else:
            self._policy_model.save_pretrained(checkpoint_path)
        
        # Save tokenizer
        self._tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_reward": self.best_reward,
            "optimizer_state": self._optimizer.state_dict(),
            "scheduler_state": self._scheduler.state_dict(),
            "train_history": self.train_history,
        }
        torch.save(state, checkpoint_path / "training_state.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints if exceeding limit."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1])
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {old_checkpoint}")
            import shutil
            shutil.rmtree(old_checkpoint)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        if self.config.use_lora and PEFT_AVAILABLE:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model_path or self.config.policy_model_name,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._policy_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            self._policy_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        
        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_reward = state["best_reward"]
            self.train_history = state["train_history"]
            
            if self._optimizer:
                self._optimizer.load_state_dict(state["optimizer_state"])
            if self._scheduler:
                self._scheduler.load_state_dict(state["scheduler_state"])
        
        logger.info(f"Resumed from step {self.global_step}")
    
    async def train(self):
        """Run the full training loop."""
        logger.info("Starting GRPO training...")
        logger.info(f"Config: {self.config}")
        
        # Log hyperparameters to TensorBoard
        if self._tb_writer is not None:
            hparams = {
                "policy_model": self.config.policy_model_name,
                "group_size": self.config.group_size,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "kl_coef": self.config.kl_coef,
                "entropy_coef": self.config.entropy_coef,
                "clip_ratio": self.config.clip_ratio,
                "use_lora": self.config.use_lora,
                "lora_rank": self.config.lora_rank if self.config.use_lora else 0,
                "num_epochs": self.config.num_epochs,
                "gradient_accumulation": self.config.gradient_accumulation_steps,
            }
            # Log as text
            hparams_text = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
            self._tb_writer.add_text("Hyperparameters", hparams_text, 0)
        
        # Setup if not already done
        if self._policy_model is None:
            self.setup()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # Training loop
        try:
            for epoch in range(self.epoch, self.config.num_epochs):
                self.epoch = epoch
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                logger.info(f"{'='*60}")
                
                epoch_metrics = await self.train_epoch()
                
                logger.info(f"\n{'='*70}")
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} Summary:")
                logger.info(f"{'='*70}")
                logger.info(f"  Total Loss:      {epoch_metrics['loss']:.6f}")
                logger.info(f"  Policy Loss:     {epoch_metrics['policy_loss']:.6f}")
                logger.info(f"  KL Loss:         {epoch_metrics['kl_loss']:.6f}")
                logger.info(f"  Entropy Loss:    {epoch_metrics['entropy_loss']:.6f}")
                logger.info(f"  Mean Reward:     {epoch_metrics['mean_reward']:.4f}")
                logger.info(f"  Acceptance Rate: {epoch_metrics['acceptance_rate']:.2%}")
                logger.info(f"{'='*70}")
                
                # TensorBoard epoch logging
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar("Epoch/loss", epoch_metrics['loss'], epoch + 1)
                    self._tb_writer.add_scalar("Epoch/policy_loss", epoch_metrics['policy_loss'], epoch + 1)
                    self._tb_writer.add_scalar("Epoch/kl_loss", epoch_metrics['kl_loss'], epoch + 1)
                    self._tb_writer.add_scalar("Epoch/entropy_loss", epoch_metrics['entropy_loss'], epoch + 1)
                    self._tb_writer.add_scalar("Epoch/mean_reward", epoch_metrics['mean_reward'], epoch + 1)
                    self._tb_writer.add_scalar("Epoch/acceptance_rate", epoch_metrics['acceptance_rate'], epoch + 1)
                    self._tb_writer.flush()
                
                # Update best reward
                if epoch_metrics["mean_reward"] > self.best_reward:
                    self.best_reward = epoch_metrics["mean_reward"]
                    self._save_checkpoint()
                    logger.info(f"New best reward: {self.best_reward:.2f}")
                
                # Check max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max steps ({self.config.max_steps})")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint()
        
        # Save final checkpoint
        self._save_checkpoint()
        
        # Save training history
        history_path = Path(self.config.log_dir) / "train_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info(f"Training complete! Final step: {self.global_step}")
        logger.info(f"Best reward: {self.best_reward:.2f}")
        
        return self.train_history
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._reward_model:
            await self._reward_model.close()
        
        # Close TensorBoard writer
        if self._tb_writer is not None:
            self._tb_writer.close()
            logger.info("TensorBoard writer closed")

