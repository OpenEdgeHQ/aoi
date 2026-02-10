# SOTA Reward Model for AOI Self-Evolve Training
"""
This module implements the Reward Model using SOTA closed-source LLMs.
The Reward Model evaluates and scores generated scenarios.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from string import Template
from dataclasses import dataclass
import yaml
import httpx

from .evolver_config import EvolverConfig, RewardConfig, DEFAULT_EVOLVER_CONFIG, DEFAULT_REWARD_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ScenarioScore:
    """
    Multi-dimensional score for a single scenario.
    
    NEW Dimensions (focused on commands quality):
    - problem_validity: Is the fault scenario realistic and reasonable?
    - commands_completeness: Does command sequence fully solve the problem? (MOST IMPORTANT)
    - commands_correctness: Are commands syntactically correct?
    - solution_effectiveness: Would these commands actually fix the problem?
    - format: Basic JSON structure correctness
    
    A good scenario needs commands that:
    1. Discover/detect the issue
    2. Diagnose the root cause
    3. FIX/RESOLVE the issue
    4. Verify the fix worked
    """
    scenario_id: int
    
    # NEW Multi-dimensional scores (0-10)
    problem_validity_score: float = 5.0      # Is fault realistic?
    commands_completeness_score: float = 5.0  # MOST IMPORTANT: Full diagnostic + solution?
    commands_correctness_score: float = 5.0   # Syntax correct?
    solution_effectiveness_score: float = 5.0 # Would it actually fix?
    format_score: float = 5.0                 # Basic structure
    
    # Aggregated score
    overall_score: float = 5.0
    
    # Legacy fields for compatibility (mapped from new scores)
    syntax_score: float = 5.0
    logic_score: float = 5.0
    diversity_score: float = 5.0
    difficulty_score: float = 5.0
    state_quality_score: float = 5.0
    logical_consistency: float = 5.0
    complexity_value: float = 5.0
    syntax_correctness: float = 5.0
    recommendation: str = "revise"
    reasoning: Dict[str, str] = None
    improvement_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = {}
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []
        # Map new scores to legacy fields for compatibility
        self.syntax_score = self.commands_correctness_score
        self.logic_score = self.solution_effectiveness_score
        self.diversity_score = self.problem_validity_score
        self.difficulty_score = self.commands_completeness_score
        self.state_quality_score = self.commands_completeness_score
        self.logical_consistency = self.solution_effectiveness_score
        self.syntax_correctness = self.commands_correctness_score
        self.complexity_value = self.commands_completeness_score
    
    @property
    def scores_dict(self) -> Dict[str, float]:
        """Get all dimension scores as dict."""
        return {
            "problem_validity": self.problem_validity_score,
            "commands_completeness": self.commands_completeness_score,
            "commands_correctness": self.commands_correctness_score,
            "solution_effectiveness": self.solution_effectiveness_score,
            "format": self.format_score,
        }
    
    @property
    def scores_list(self) -> List[float]:
        """Get all dimension scores as list."""
        return [
            self.problem_validity_score,
            self.commands_completeness_score,
            self.commands_correctness_score,
            self.solution_effectiveness_score,
            self.format_score,
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            # New dimensions
            "problem_validity_score": self.problem_validity_score,
            "commands_completeness_score": self.commands_completeness_score,
            "commands_correctness_score": self.commands_correctness_score,
            "solution_effectiveness_score": self.solution_effectiveness_score,
            "format_score": self.format_score,
            "overall_score": self.overall_score,
            # Legacy (for compatibility)
            "logical_consistency": self.logical_consistency,
            "complexity_value": self.complexity_value,
            "syntax_correctness": self.syntax_correctness,
            "recommendation": self.recommendation,
        }


class RewardModel:
    """
    SOTA Reward Model for evaluating generated scenarios.
    
    Uses a high-quality closed-source LLM (e.g., Claude Sonnet) to evaluate
    scenarios across multiple dimensions and provide scalar reward values
    for GRPO/PPO training.
    """
    
    def __init__(
        self,
        evolver_config: Optional[EvolverConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        """
        Initialize the Reward Model.
        
        Args:
            evolver_config: Configuration for LLM API access
            reward_config: Configuration for reward scoring
        """
        self.evolver_config = evolver_config or DEFAULT_EVOLVER_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self._prompts = self._load_prompts()
        self._client = None
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load reward prompts from YAML file."""
        prompts_path = Path(__file__).parent / "prompts" / "reward_prompts.yaml"
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Call the LLM API for evaluation.
        
        Args:
            messages: List of message dicts
            temperature: Override temperature
            model: Override model
            use_cache: Whether to use OpenRouter/Anthropic prompt caching
            
        Returns:
            LLM response text
        """
        client = self._get_client()
        model_name = model or self.evolver_config.reward_model
        
        headers = {
            "Authorization": f"Bearer {self.evolver_config.api_key}",
            "Content-Type": "application/json",
        }
        
        if "openrouter" in self.evolver_config.api_base:
            headers["HTTP-Referer"] = "https://github.com/aoi"
            headers["X-Title"] = "AOI Reward Model"
        
        # Apply Anthropic native prompt caching for claude models
        # This caches the system prompt to reduce costs (up to 90% savings)
        cached_messages = messages
        is_anthropic_model = "anthropic" in model_name.lower() or "claude" in model_name.lower()
        
        if use_cache and is_anthropic_model and "openrouter" in self.evolver_config.api_base:
            # Anthropic prompt caching: add cache_control to system message
            # The system prompt is static and can be cached across requests
            cached_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    # Add cache_control to cache the system prompt
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
            logger.debug(f"Enabled Anthropic prompt caching for model: {model_name}")
        
        payload = {
            "model": model_name,
            "messages": cached_messages,
            "temperature": temperature if temperature is not None else self.evolver_config.reward_temperature,
            "max_tokens": 4096,
        }
        
        # Enable OpenRouter provider settings and transforms
        if use_cache and "openrouter" in self.evolver_config.api_base:
            payload["provider"] = {
                "allow_fallbacks": True,
                "require_parameters": False,
            }
            # middle-out is still useful for non-Anthropic models
            if not is_anthropic_model:
                payload["transforms"] = ["middle-out"]
        
        url = f"{self.evolver_config.api_base}/chat/completions"
        
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Log the raw API response structure for debugging
            logger.debug(f"Reward Model API response status: {response.status_code}")
            
            # Log cache usage info if available (OpenRouter returns this in response)
            usage = result.get("usage", {})
            if usage:
                cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
                cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                if cache_creation_tokens > 0 or cache_read_tokens > 0:
                    logger.debug(
                        f"Prompt Caching: created={cache_creation_tokens}, "
                        f"read={cache_read_tokens}, prompt={prompt_tokens}, "
                        f"completion={completion_tokens}"
                    )
                    # Cache read = cost savings!
                    if cache_read_tokens > 0:
                        savings_pct = (cache_read_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0
                        logger.debug(f"Cache savings: ~{savings_pct:.1f}% of input tokens from cache")
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Log the extracted content (debug level)
            logger.debug(f"Reward Model response content (first 300 chars): {content[:300] if content else 'EMPTY'}...")
            
            if not content:
                logger.error(f"No content in API response. Full response: {json.dumps(result, indent=2)[:500]}")
            
            return content
        except Exception as e:
            logger.error(f"Reward Model LLM API call failed: {e}")
            raise
    
    def _calculate_weighted_score(
        self,
        logical: float,
        complexity: float,
        syntax: float,
    ) -> float:
        """Calculate weighted overall score (legacy 3-dimension)."""
        return (
            logical * self.reward_config.logical_consistency_weight +
            complexity * self.reward_config.complexity_weight +
            syntax * self.reward_config.syntax_correctness_weight
        )
    
    def _calculate_multidim_score(
        self,
        problem_validity: float,
        commands_completeness: float,
        commands_correctness: float,
        solution_effectiveness: float,
        format_score: float = 10.0,  # Default to full score (JSON parsed = format OK)
        diversity_score: float = 10.0,  # Default to full score, caller should provide actual value
    ) -> float:
        """
        Calculate weighted overall score from multi-dimensional rewards.
        
        WEIGHTS (total = 1.0):
        - solution_effectiveness: 0.30 (Does the solution actually fix the problem?)
        - commands_completeness: 0.20 (Are all diagnostic + resolution steps included?)
        - diversity: 0.20 (Different from seed, anti-plagiarism)
        - problem_validity: 0.10 (Is the fault scenario realistic?)
        - commands_correctness: 0.10 (Are commands syntactically correct?)
        - format: 0.10 (JSON parsing success)
        
        NOTE:
        - format_score is determined by code (JSON parseable = 10)
        - diversity_score is determined by code (by default 10.0 in offline scoring; training uses a separate diversity reward)
        """
        weights = {
            "solution_effectiveness": 0.30,  # MOST IMPORTANT
            "commands_completeness": 0.20,   # Second most important
            "diversity": 0.20,               # Anti-plagiarism
            "problem_validity": 0.10,
            "commands_correctness": 0.10,
            "format": 0.10,
        }
        
        return (
            solution_effectiveness * weights["solution_effectiveness"] +
            commands_completeness * weights["commands_completeness"] +
            diversity_score * weights["diversity"] +
            problem_validity * weights["problem_validity"] +
            commands_correctness * weights["commands_correctness"] +
            format_score * weights["format"]
        )
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score from 0-10 to 0-1 range."""
        return max(0.0, min(1.0, score / 10.0))
    
    def _extract_json(self, response: str) -> Optional[str]:
        """
        Extract JSON from LLM response.
        
        Handles various formats:
        - Pure JSON
        - JSON in code blocks (````json ... ```` or ```json ... ``` etc.)
        - JSON preceded by text explanations
        """
        if not response:
            return None
        
        response = response.strip()
        
        # Try to find JSON in code blocks first (4 backticks have priority)
        if "````json" in response:
            json_start = response.find("````json") + 8
            json_end = response.find("````", json_start)
            if json_end > json_start:
                return response[json_start:json_end].strip()
        
        if "````" in response:
            json_start = response.find("````") + 4
            json_end = response.find("````", json_start)
            if json_end > json_start:
                extracted = response[json_start:json_end].strip()
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted
        
        # Try 3 backticks
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            if json_end > json_start:
                return response[json_start:json_end].strip()
        
        if "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            if json_end > json_start:
                extracted = response[json_start:json_end].strip()
                # Make sure it looks like JSON
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted
        
        # Find JSON object directly in the response
        brace_start = response.find("{")
        bracket_start = response.find("[")
        
        if brace_start >= 0 or bracket_start >= 0:
            # Use whichever comes first
            if brace_start >= 0 and (bracket_start < 0 or brace_start < bracket_start):
                # Find matching closing brace
                start = brace_start
                depth = 0
                for i, char in enumerate(response[start:], start):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            return response[start:i+1]
            elif bracket_start >= 0:
                # Find matching closing bracket
                start = bracket_start
                depth = 0
                for i, char in enumerate(response[start:], start):
                    if char == "[":
                        depth += 1
                    elif char == "]":
                        depth -= 1
                        if depth == 0:
                            return response[start:i+1]
        
        # If nothing found, return the whole response (might still be valid JSON)
        return response
    
    async def score_single(
        self,
        candidate: Dict[str, Any],
    ) -> ScenarioScore:
        """
        Score a single candidate scenario.
        
        Args:
            candidate: The candidate scenario to evaluate
            
        Returns:
            ScenarioScore with detailed scores
        """
        system_prompt = self._prompts["system_prompt"]
        scoring_template = Template(self._prompts["scoring_prompt"])
        
        # Extract fields from candidate (new simplified format)
        # Required fields: problem_id, task_description, system_state_summary, commands
        problem_id = candidate.get("problem_id", "unknown")
        task_description = candidate.get("task_description", "N/A")
        system_state_summary = candidate.get("system_state_summary", "N/A")
        commands = candidate.get("commands", candidate.get("command_list", []))
        
        user_prompt = scoring_template.substitute(
            problem_id=problem_id,
            task_description=task_description,
            system_state_summary=system_state_summary,
            commands="\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(commands)),
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await self._call_llm(messages)
            
            # Log raw response for debugging
            # logger.info(f"Reward model raw response: {response[:300] if response else 'EMPTY'}...")
            
            if not response or not response.strip():
                logger.error("Reward model returned empty response")
                raise ValueError("Empty response from reward model")
            
            # Extract JSON from response
            json_text = self._extract_json(response)
            
            if not json_text:
                logger.error(f"Could not extract JSON from response: {response[:200]}")
                raise ValueError("No JSON found in response")
            
            eval_result = json.loads(json_text)
            
            # Parse NEW multi-dimensional scores (4 dimensions from LLM)
            # New format: {"problem_validity": N, "commands_completeness": N, "commands_correctness": N, "solution_effectiveness": N}
            # NO FALLBACK: if a dimension is missing, score is 0
            # format_score is determined by code (JSON parseable = 10), not LLM
            
            problem_validity = float(eval_result.get("problem_validity", 0))
            commands_completeness = float(eval_result.get("commands_completeness", 0))
            commands_correctness = float(eval_result.get("commands_correctness", 0))
            solution_effectiveness = float(eval_result.get("solution_effectiveness", 0))
            # format_score: If we got here, JSON was parseable -> format is correct
            format_score = 10.0  # Full score for valid JSON
            
            # Calculate weighted overall score with NEW weights
            overall = self._calculate_multidim_score(
                problem_validity, commands_completeness, commands_correctness, 
                solution_effectiveness, format_score
            )
            
            logger.debug(
                f"Scores: problem_validity={problem_validity:.1f}, "
                f"commands_completeness={commands_completeness:.1f}, "
                f"commands_correctness={commands_correctness:.1f}, "
                f"solution_effectiveness={solution_effectiveness:.1f}, "
                f"format={format_score:.1f} (auto), overall={overall:.2f}"
            )
            
            return ScenarioScore(
                scenario_id=candidate.get("candidate_id", 0),
                problem_validity_score=problem_validity,
                commands_completeness_score=commands_completeness,
                commands_correctness_score=commands_correctness,
                solution_effectiveness_score=solution_effectiveness,
                format_score=format_score,
                overall_score=overall,
                recommendation="accept" if overall >= 7 else ("revise" if overall >= 4 else "reject"),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reward model response: {e}")
            logger.error(f"Raw response was: {response[:500] if response else 'EMPTY'}...")
            logger.error(f"Extracted JSON text was: {json_text[:500] if json_text else 'EMPTY'}...")
            # NO FALLBACK: Return 0 scores on parse failure
            return ScenarioScore(
                scenario_id=candidate.get("candidate_id", 0),
                problem_validity_score=0.0,
                commands_completeness_score=0.0,
                commands_correctness_score=0.0,
                solution_effectiveness_score=0.0,
                format_score=0.0,
                overall_score=0.0,
                recommendation="reject",
                reasoning={"error": f"Failed to parse evaluation response: {e}"},
            )
        except Exception as e:
            logger.error(f"Failed to score candidate: {e}")
            raise
    
    async def score_batch(
        self,
        candidates: List[Dict[str, Any]],
        seed_summary: Optional[str] = None,
    ) -> List[ScenarioScore]:
        """
        Score multiple candidates in batch.
        
        Args:
            candidates: List of candidate scenarios
            seed_summary: Optional summary of the original seed
            
        Returns:
            List of ScenarioScore objects
        """
        # Score each candidate in parallel
        tasks = [self.score_single(candidate) for candidate in candidates]
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        valid_scores = []
        for i, score in enumerate(scores):
            if isinstance(score, Exception):
                logger.warning(f"Scoring candidate {i} failed: {score}")
                # NO FALLBACK: return 0 scores on failure
                valid_scores.append(ScenarioScore(
                    scenario_id=i,
                    problem_validity_score=0.0,
                    commands_completeness_score=0.0,
                    commands_correctness_score=0.0,
                    solution_effectiveness_score=0.0,
                    format_score=0.0,
                    overall_score=0.0,
                    recommendation="reject",
                    reasoning={"error": str(score)},
                    improvement_suggestions=[],
                ))
            else:
                valid_scores.append(score)
        
        return valid_scores
    
    # NOTE: score_batch_single_call removed - each candidate is now scored individually
    # via score_single() for better accuracy and prompt caching efficiency
    
    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[ScenarioScore],
    ) -> List[Tuple[Dict[str, Any], ScenarioScore]]:
        """
        Rank candidates by their scores.
        
        Args:
            candidates: List of candidate scenarios
            scores: List of corresponding scores
            
        Returns:
            List of (candidate, score) tuples sorted by overall_score descending
        """
        # Pair candidates with scores
        paired = list(zip(candidates, scores))
        
        # Sort by overall score descending
        paired.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return paired
    
    def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[ScenarioScore],
        min_score: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], ScenarioScore]]:
        """
        Filter candidates by minimum score threshold.
        
        Args:
            candidates: List of candidate scenarios
            scores: List of corresponding scores
            min_score: Minimum overall score (default: config value)
            
        Returns:
            List of (candidate, score) tuples above threshold
        """
        threshold = min_score if min_score is not None else self.reward_config.min_acceptable_score * 10
        
        filtered = [
            (c, s) for c, s in zip(candidates, scores)
            if s.overall_score >= threshold
        ]
        
        return filtered
    
    def compute_grpo_advantages(
        self,
        scores: List[ScenarioScore],
    ) -> List[float]:
        """
        Compute GRPO advantages for a group of candidates.
        
        GRPO computes advantage as: A_i = (score_i - mean(scores)) / std(scores)
        
        Args:
            scores: List of scenario scores
            
        Returns:
            List of advantage values (same order as input)
        """
        if not scores:
            return []
        
        # Extract overall scores
        score_values = [s.overall_score for s in scores]
        
        # Compute mean and std
        mean_score = sum(score_values) / len(score_values)
        variance = sum((s - mean_score) ** 2 for s in score_values) / len(score_values)
        std_score = variance ** 0.5 if variance > 0 else 1.0
        
        # Compute advantages
        advantages = [(s - mean_score) / std_score for s in score_values]
        
        return advantages
    
    def select_best_candidates(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[ScenarioScore],
        top_k: int = 1,
    ) -> List[Tuple[Dict[str, Any], ScenarioScore]]:
        """
        Select the top-k candidates by score.
        
        Args:
            candidates: List of candidate scenarios
            scores: List of corresponding scores
            top_k: Number of candidates to select
            
        Returns:
            List of (candidate, score) tuples for top-k candidates
        """
        ranked = self.rank_candidates(candidates, scores)
        return ranked[:top_k]
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Synchronous wrapper
class RewardModelSync:
    """Synchronous wrapper for RewardModel."""
    
    def __init__(
        self,
        evolver_config: Optional[EvolverConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        self._model = RewardModel(evolver_config, reward_config)
    
    def score_single(self, candidate: Dict[str, Any]) -> ScenarioScore:
        """Score single candidate synchronously."""
        return asyncio.run(self._model.score_single(candidate))
    
    def score_batch(
        self,
        candidates: List[Dict[str, Any]],
        seed_summary: Optional[str] = None,
    ) -> List[ScenarioScore]:
        """Score batch synchronously."""
        return asyncio.run(self._model.score_batch(candidates, seed_summary))
    
    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[ScenarioScore],
    ) -> List[Tuple[Dict[str, Any], ScenarioScore]]:
        """Rank candidates."""
        return self._model.rank_candidates(candidates, scores)
    
    def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[ScenarioScore],
        min_score: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], ScenarioScore]]:
        """Filter candidates."""
        return self._model.filter_candidates(candidates, scores, min_score)
    
    def compute_grpo_advantages(self, scores: List[ScenarioScore]) -> List[float]:
        """Compute GRPO advantages."""
        return self._model.compute_grpo_advantages(scores)

