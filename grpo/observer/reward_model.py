# Observer Reward Model for GRPO Training
"""
Multi-dimensional reward model for Observer GRPO training.
Uses Claude Sonnet via OpenRouter for LLM-based evaluation.

Dimensions:
- format (0.10): Rule-based JSON parsing check
- summary (0.15): LLM-evaluated previous_iteration_summary quality
- action (0.10): LLM-evaluated next_action correctness
- context_instruction (0.30): LLM-evaluated probe/executor context quality
- context_namespace (0.30): LLM-evaluated target resource accuracy
- confidence (0.05): LLM-evaluated confidence calibration
"""

import json
import logging
import asyncio
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from string import Template
import yaml
import httpx

from .grpo_config import ObserverGRPOConfig, RewardWeights

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score for a single dimension."""
    name: str
    raw_score: float  # 0-10 scale
    weight: float
    weighted_score: float  # raw_score / 10 * weight
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "raw_score": self.raw_score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
        }


@dataclass
class RewardScore:
    """Complete reward score for a candidate."""
    candidate_id: int
    
    # Dimension scores
    format_score: DimensionScore = None
    summary_score: DimensionScore = None
    action_score: DimensionScore = None
    context_instruction_score: DimensionScore = None
    context_namespace_score: DimensionScore = None
    confidence_score: DimensionScore = None
    
    # Total score
    total_score: float = 0.0
    
    # Metadata
    skip_summary: bool = False  # True when n=1 or n=m-1
    is_submit_stage: bool = False  # True when n=m-1
    submit_matched: bool = False  # True when submit command matches GT
    json_parse_success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        scores = {}
        for name in ["format", "summary", "action", "context_instruction", "context_namespace", "confidence"]:
            score = getattr(self, f"{name}_score")
            if score:
                scores[name] = score.to_dict()
        
        return {
            "candidate_id": self.candidate_id,
            "scores": scores,
            "total_score": self.total_score,
            "skip_summary": self.skip_summary,
            "is_submit_stage": self.is_submit_stage,
            "submit_matched": self.submit_matched,
            "json_parse_success": self.json_parse_success,
        }


def normalize_submit_command(cmd: str) -> str:
    """Normalize submit command for comparison (remove escapes, whitespace)."""
    if not cmd:
        return ""
    # Remove backslashes
    cmd = cmd.replace("\\", "")
    # Remove extra whitespace
    cmd = " ".join(cmd.split())
    return cmd


def parse_json_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model output (first valid JSON only)."""
    result = parse_json_output_with_count(text)
    return result[0] if result else None


def parse_json_output_with_count(text: str) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    Parse JSON from model output and count JSON blocks.
    
    Returns:
        Tuple of (parsed_json, json_block_count) or None if parse failed.
        json_block_count: 1 = single clean output, >1 = multiple/repeated outputs
    """
    if not text:
        return None
    
    original_text = text.strip()
    text = original_text
    
    # Count code blocks (```json or ```)
    code_block_count = 0
    for marker in ["```json", "```"]:
        code_block_count += text.count(marker)
    # Each block has open and close, so divide by 2
    code_block_count = code_block_count // 2
    
    # Try code blocks first
    for marker in ["````json", "````", "```json", "```"]:
        if marker in text:
            start = text.find(marker) + len(marker)
            end_marker = "````" if marker.startswith("````") else "```"
            end = text.find(end_marker, start)
            if end > start:
                text = text[start:end].strip()
                break
    
    # Find first JSON object
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
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    
    # Count JSON objects in original text (look for multiple top-level braces)
    # This detects repeated JSON outputs even without code blocks
    json_object_count = 0
    remaining = original_text
    while True:
        start_idx = remaining.find("{")
        if start_idx < 0:
            break
        
        depth = 0
        found_end = False
        for i, char in enumerate(remaining[start_idx:], start_idx):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_object_count += 1
                    remaining = remaining[i+1:]
                    found_end = True
                    break
        
        if not found_end:
            break
    
    # Use max of code block count and json object count
    block_count = max(code_block_count, json_object_count, 1)
    
    return (parsed, block_count)


class ObserverRewardModel:
    """
    Multi-dimensional reward model for Observer GRPO training.
    """
    
    def __init__(self, config: ObserverGRPOConfig):
        self.config = config
        self.weights = config.reward_weights
        self._client = None
        self._prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load reward prompts from YAML."""
        prompts_path = Path(self.config.reward_prompts_path)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with proxy."""
        if self._client is None:
            if self.config.proxy_url:
                self._client = httpx.AsyncClient(
                    timeout=120.0,
                    proxy=self.config.proxy_url,
                )
            else:
                self._client = httpx.AsyncClient(timeout=120.0)
        return self._client
    
    async def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
        use_cache: bool = True,
    ) -> str:
        """
        Call the LLM API with infinite retry on failure.
        Reference: observer/observer_reward_model.py
        """
        client = self._get_client()
        model_name = self.config.reward_model_name
        api_base = self.config.reward_model_api_base
        
        headers = {
            "Authorization": f"Bearer {self.config.reward_model_api_key}",
            "Content-Type": "application/json",
        }
        
        # OpenRouter specific headers (for usage tracking)
        if "openrouter" in api_base:
            headers["HTTP-Referer"] = "https://aoi-observer-grpo"
            headers["X-Title"] = "AOI Observer GRPO Training"
        
        # Apply Anthropic prompt caching only for Claude models on OpenRouter
        is_anthropic_model = "anthropic" in model_name.lower() or "claude" in model_name.lower()
        cached_messages = messages
        
        if use_cache and is_anthropic_model and "openrouter" in api_base:
            cached_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    # Cache system prompt
                    cached_msg = {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    }
                    cached_messages.append(cached_msg)
                else:
                    cached_messages.append(msg)
        
        payload = {
            "model": model_name,
            "messages": cached_messages,
            "temperature": self.config.reward_temperature,
            "max_tokens": 1024,
        }
        
        # OpenRouter provider settings
        if "openrouter" in api_base:
            payload["provider"] = {
                "allow_fallbacks": False,  # Fail fast for debugging
                "require_parameters": False,
            }
        
        url = f"{api_base}/chat/completions"
        retry_delay = self.config.retry_interval
        attempt = 0
        
        # Infinite retry with interval
        while True:
            attempt += 1
            try:
                response = await client.post(url, headers=headers, json=payload)
                
                # Handle retryable errors
                if response.status_code in (502, 503, 504, 429):
                    logger.warning(f"API Error (HTTP {response.status_code}) - Attempt {attempt}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                response.raise_for_status()
                
                if attempt > 1:
                    logger.info(f"API connection restored after {attempt} attempts")
                
                result = response.json()
                
                # Log cache usage
                usage = result.get("usage", {})
                cache_read = usage.get("cache_read_input_tokens", 0)
                if cache_read > 0:
                    logger.debug(f"Cache hit: {cache_read} tokens from cache")
                
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not content:
                    logger.warning("Empty response from reward model, retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Delay after successful request to avoid rate limiting (OpenRouter Claude limit)
                await asyncio.sleep(3)
                
                return content
                
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error {e.response.status_code}: {e.response.text[:200]}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                logger.warning(f"Connection error ({type(e).__name__}) - Attempt {attempt}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.warning(f"Reward API call failed: {e}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
    
    def _compute_weights(self, skip_summary: bool) -> Dict[str, float]:
        """Get weights, optionally redistributing summary weight."""
        if skip_summary:
            return self.weights.get_weights_without_summary()
        return self.weights.get_all_weights()
    
    def _create_dimension_score(
        self,
        name: str,
        raw_score: float,
        weight: float,
    ) -> DimensionScore:
        """Create a dimension score."""
        return DimensionScore(
            name=name,
            raw_score=raw_score,
            weight=weight,
            weighted_score=(raw_score / 10.0) * weight,
        )
    
    async def score_single(
        self,
        candidate_output: str,
        problem_id: str,
        task_type: str,
        task_description: str,
        current_iter: int,
        total_iters: int,
        expected_action: str,
        execution_history: str,
        steps: List[Dict[str, Any]],
        candidate_id: int = 0,
    ) -> RewardScore:
        """
        Score a single candidate output.
        
        Args:
            candidate_output: Raw model output string
            problem_id: Problem ID
            task_type: Task type (detection, localization, analysis, mitigation)
            task_description: Task description
            current_iter: Current iteration (n)
            total_iters: Total iterations (m)
            expected_action: Expected action type
            execution_history: Full execution history for reward model
            steps: List of step dicts from data file
            candidate_id: Candidate index
        
        Returns:
            RewardScore with all dimension scores
        """
        result = RewardScore(candidate_id=candidate_id)
        
        # Determine if this is submit stage - based on expected_action, not iteration number
        # This is because some tasks have probe at n=m-1, submit stage is only when expected is submit
        is_submit_stage = (expected_action == "submit")
        result.is_submit_stage = is_submit_stage
        
        # Determine if we should skip summary (n=1, or submit stage)
        skip_summary = (current_iter == 1 or is_submit_stage)
        result.skip_summary = skip_summary
        
        # Get weights
        weights = self._compute_weights(skip_summary)
        
        # Step 1: Format check (rule-based)
        # Use parse_json_output_with_count to detect multiple JSON blocks
        parse_result = parse_json_output_with_count(candidate_output)
        
        if parse_result is None:
            # JSON parse failed - apply penalty
            result.json_parse_success = False
            result.format_score = self._create_dimension_score("format", 0.0, weights.get("format", 0.10))
            
            # All other dimensions get 0.9 (0.09 * 10) as penalty
            penalty_raw = 0.9
            if not skip_summary:
                result.summary_score = self._create_dimension_score("summary", penalty_raw, weights.get("summary", 0.15))
            result.action_score = self._create_dimension_score("action", penalty_raw, weights.get("action", 0.10))
            result.context_instruction_score = self._create_dimension_score("context_instruction", penalty_raw, weights.get("context_instruction", 0.30))
            result.context_namespace_score = self._create_dimension_score("context_namespace", penalty_raw, weights.get("context_namespace", 0.30))
            result.confidence_score = self._create_dimension_score("confidence", penalty_raw, weights.get("confidence", 0.05))
            
            # Total = 0.09
            result.total_score = 0.09
            return result
        
        # JSON parsed successfully
        parsed, json_block_count = parse_result
        result.json_parse_success = True
        
        # Format score: 10 for single clean output, 5 for multiple/repeated outputs
        if json_block_count == 1:
            format_raw_score = 10.0  # Clean single JSON output
        else:
            format_raw_score = 5.0   # Multiple JSON blocks detected, penalize
            logger.debug(f"Multiple JSON blocks detected ({json_block_count}), format score reduced to 5.0")
        
        result.format_score = self._create_dimension_score("format", format_raw_score, weights.get("format", 0.10))
        
        # Step 2: Check if this is submit stage with special handling
        if is_submit_stage:
            return await self._score_submit_stage(
                result, parsed, steps, current_iter, total_iters, weights
            )
        
        # Step 3: Normal LLM-based evaluation
        return await self._score_with_llm(
            result, parsed, candidate_output,
            problem_id, task_type, task_description,
            current_iter, total_iters, expected_action,
            execution_history, weights, skip_summary
        )
    
    async def _score_submit_stage(
        self,
        result: RewardScore,
        parsed: Dict[str, Any],
        steps: List[Dict[str, Any]],
        current_iter: int,
        total_iters: int,
        weights: Dict[str, float],
    ) -> RewardScore:
        """Score submit stage using rule-based evaluation."""
        
        # Get ground truth submit command
        gt_submit_command = steps[total_iters - 1].get("command", "") if total_iters <= len(steps) else ""
        
        # Get candidate submit command
        submission = parsed.get("submission", {})
        candidate_command = submission.get("submission_command", "")
        
        # Check action
        next_action = parsed.get("next_action", {})
        candidate_action = next_action.get("action", "")
        
        # Get confidence
        confidence = parsed.get("confidence", 0)
        
        # Compare submit commands
        gt_normalized = normalize_submit_command(gt_submit_command)
        candidate_normalized = normalize_submit_command(candidate_command)
        
        submit_matched = (gt_normalized == candidate_normalized)
        result.submit_matched = submit_matched
        
        if submit_matched:
            # All dimensions get full score (except confidence has special rules)
            result.action_score = self._create_dimension_score("action", 10.0, weights.get("action", 0.12))
            result.context_instruction_score = self._create_dimension_score("context_instruction", 10.0, weights.get("context_instruction", 0.35))
            result.context_namespace_score = self._create_dimension_score("context_namespace", 10.0, weights.get("context_namespace", 0.35))
            
            # Confidence scoring
            if confidence == 100:
                conf_raw = 10.0
            elif confidence == 99:
                conf_raw = 8.0
            elif confidence == 98:
                conf_raw = 6.0
            else:
                conf_raw = 0.0
            result.confidence_score = self._create_dimension_score("confidence", conf_raw, weights.get("confidence", 0.08))
        else:
            # Submit not matched
            # Action: check if it's submit
            if candidate_action == "submit":
                action_raw = 5.0  # Is submit but wrong content
            else:
                action_raw = 0.0  # Not submit
            result.action_score = self._create_dimension_score("action", action_raw, weights.get("action", 0.12))
            
            # Confidence by rules
            if confidence == 100:
                conf_raw = 10.0
            elif confidence == 99:
                conf_raw = 8.0
            elif confidence == 98:
                conf_raw = 6.0
            else:
                conf_raw = 0.0
            result.confidence_score = self._create_dimension_score("confidence", conf_raw, weights.get("confidence", 0.08))
            
            # Other dimensions: 0
            result.context_instruction_score = self._create_dimension_score("context_instruction", 0.0, weights.get("context_instruction", 0.35))
            result.context_namespace_score = self._create_dimension_score("context_namespace", 0.0, weights.get("context_namespace", 0.35))
        
        # Calculate total
        result.total_score = (
            result.format_score.weighted_score +
            result.action_score.weighted_score +
            result.context_instruction_score.weighted_score +
            result.context_namespace_score.weighted_score +
            result.confidence_score.weighted_score
        )
        
        return result
    
    async def _score_with_llm(
        self,
        result: RewardScore,
        parsed: Dict[str, Any],
        candidate_output: str,
        problem_id: str,
        task_type: str,
        task_description: str,
        current_iter: int,
        total_iters: int,
        expected_action: str,
        execution_history: str,
        weights: Dict[str, float],
        skip_summary: bool,
    ) -> RewardScore:
        """Score using LLM evaluation."""
        
        # Build reward prompt
        system_prompt = self._prompts.get("system_prompt", "")
        scoring_template = Template(self._prompts.get("scoring_prompt", ""))
        
        user_prompt = scoring_template.safe_substitute(
            problem_id=problem_id,
            task_type=task_type,
            task_description=task_description,
            current_iteration=current_iter,
            total_iterations=total_iters,
            expected_action=expected_action,
            execution_history=execution_history,
            candidate_output=candidate_output,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Call LLM
        response = await self._call_llm_with_retry(messages)
        
        # Parse LLM response
        llm_scores = parse_json_output(response)
        
        if llm_scores is None:
            # LLM evaluation failed - use default scores
            logger.warning(f"Failed to parse LLM reward response: {response[:200]}")
            llm_scores = {
                "summary": 5.0,
                "action": 5.0,
                "context_instruction": 5.0,
                "context_namespace": 5.0,
                "confidence": 5.0,
            }
        
        # Create dimension scores
        if not skip_summary:
            result.summary_score = self._create_dimension_score(
                "summary", 
                float(llm_scores.get("summary", 5.0)),
                weights.get("summary", 0.15)
            )
        
        result.action_score = self._create_dimension_score(
            "action",
            float(llm_scores.get("action", 5.0)),
            weights.get("action", 0.10)
        )
        
        result.context_instruction_score = self._create_dimension_score(
            "context_instruction",
            float(llm_scores.get("context_instruction", 5.0)),
            weights.get("context_instruction", 0.30)
        )
        
        result.context_namespace_score = self._create_dimension_score(
            "context_namespace",
            float(llm_scores.get("context_namespace", 5.0)),
            weights.get("context_namespace", 0.30)
        )
        
        result.confidence_score = self._create_dimension_score(
            "confidence",
            float(llm_scores.get("confidence", 5.0)),
            weights.get("confidence", 0.05)
        )
        
        # Calculate total
        total = result.format_score.weighted_score
        if result.summary_score:
            total += result.summary_score.weighted_score
        total += result.action_score.weighted_score
        total += result.context_instruction_score.weighted_score
        total += result.context_namespace_score.weighted_score
        total += result.confidence_score.weighted_score
        
        result.total_score = total
        
        return result
    
    async def score_batch(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> List[RewardScore]:
        """
        Score a batch of candidates sequentially (to avoid API rate limiting).
        
        Args:
            candidates: List of (candidate_output, sample_info) tuples
                sample_info should contain: problem_id, task_type, task_description,
                current_iter, total_iters, expected_action, execution_history, steps
        
        Returns:
            List of RewardScore objects
        """
        results = []
        
        for i, (output, info) in enumerate(candidates):
            try:
                score = await self.score_single(
                    candidate_output=output,
                    problem_id=info.get("problem_id", ""),
                    task_type=info.get("task_type", ""),
                    task_description=info.get("task_description", ""),
                    current_iter=info.get("current_iter", 1),
                    total_iters=info.get("total_iters", 1),
                    expected_action=info.get("expected_action", "probe"),
                    execution_history=info.get("execution_history", ""),
                    steps=info.get("steps", []),
                    candidate_id=i,
                )
                results.append(score)
            except Exception as e:
                logger.error(f"Scoring candidate {i} failed: {e}")
                # Return minimum score
                results.append(RewardScore(candidate_id=i, total_score=0.09))
        
        return results
    
    def compute_advantages(self, scores: List[RewardScore]) -> List[float]:
        """
        Compute GRPO advantages for a group of candidates.
        
        A_i = (score_i - mean(scores)) / std(scores)
        """
        if not scores:
            return []
        
        score_values = [s.total_score for s in scores]
        
        mean_score = sum(score_values) / len(score_values)
        variance = sum((s - mean_score) ** 2 for s in score_values) / len(score_values)
        std_score = variance ** 0.5 if variance > 0 else 1.0
        
        advantages = [(s - mean_score) / std_score for s in score_values]
        
        return advantages
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class RewardModelSync:
    """Synchronous wrapper for ObserverRewardModel."""
    
    def __init__(self, config: ObserverGRPOConfig):
        self._model = ObserverRewardModel(config)
        self._loop = None
    
    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop
    
    def score_single(self, **kwargs) -> RewardScore:
        """Score single candidate synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self._model.score_single(**kwargs))
    
    def score_batch(self, candidates: List[Tuple[str, Dict[str, Any]]]) -> List[RewardScore]:
        """Score batch synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self._model.score_batch(candidates))
    
    def compute_advantages(self, scores: List[RewardScore]) -> List[float]:
        """Compute advantages."""
        return self._model.compute_advantages(scores)
