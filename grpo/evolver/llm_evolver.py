# LLM-based Evolver for AOI Self-Evolve Training
"""
This module implements the Evolver component using closed-source LLMs.
The Evolver generates diverse fault scenarios from seed data.
"""

import json
import logging
import random
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from string import Template
import yaml
import httpx

from .evolver_config import EvolverConfig, DEFAULT_EVOLVER_CONFIG

logger = logging.getLogger(__name__)


class LLMEvolver:
    """
    LLM-based Evolver for generating diverse fault scenarios.
    
    This evolver uses closed-source LLMs (via OpenRouter/OpenAI API) to generate
    multiple candidate scenarios from a seed scenario, implementing the "Policy Model"
    role in the GRPO/PPO training framework.
    """
    
    def __init__(self, config: Optional[EvolverConfig] = None):
        """
        Initialize the LLM Evolver.
        
        Args:
            config: Evolver configuration. Uses default if not provided.
        """
        self.config = config or DEFAULT_EVOLVER_CONFIG
        self._prompts = self._load_prompts()
        self._client = None
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        prompts_path = Path(__file__).parent / "prompts" / "evolver_prompts.yaml"
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
        Call the LLM API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override temperature for this call
            model: Override model for this call
            use_cache: Whether to use OpenRouter prompt caching
            
        Returns:
            LLM response text
        """
        client = self._get_client()
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add OpenRouter specific headers
        if "openrouter" in self.config.api_base:
            headers["HTTP-Referer"] = "https://github.com/aoi"
            headers["X-Title"] = "AOI Evolver"
        
        payload = {
            "model": model or self.config.evolver_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.evolver_temperature,
            "max_tokens": 8192,
        }
        
        # Enable OpenRouter prompt caching
        if use_cache and "openrouter" in self.config.api_base:
            # Add cache control to system message for OpenRouter
            if messages and messages[0].get("role") == "system":
                # OpenRouter uses provider routing for caching
                payload["provider"] = {
                    "allow_fallbacks": True,
                    "require_parameters": False,
                }
                # Enable prompt caching via transforms
                payload["transforms"] = ["middle-out"]
        
        url = f"{self.config.api_base}/chat/completions"
        
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_seed_data(self, seed_path: str) -> Dict[str, Any]:
        """
        Parse seed data from JSON file.
        
        Args:
            seed_path: Path to the seed JSON file
            
        Returns:
            Parsed seed data
        """
        with open(seed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "task_info": data.get("task_info", {}),
            "commands": data.get("commands", []),
            "evaluation_results": data.get("evaluation_results", {}),
        }
    
    def _detect_task_type(self, seed_data: Dict[str, Any]) -> str:
        """
        Detect the task type from seed data.
        
        Args:
            seed_data: Parsed seed data
            
        Returns:
            Task type: 'detection', 'localization', 'analysis', or 'mitigation'
        """
        eval_results = seed_data.get("evaluation_results", {})
        task_desc = seed_data.get("task_info", {}).get("task_description", "")
        problem_id = seed_data.get("task_info", {}).get("problem_id", "")
        
        # Check problem_id first (most reliable)
        if "-detection-" in problem_id:
            return "detection"
        elif "-localization-" in problem_id:
            return "localization"
        elif "-analysis-" in problem_id:
            return "analysis"
        elif "-mitigation-" in problem_id:
            return "mitigation"
        
        # Fallback to evaluation results keys
        if "Detection Accuracy" in eval_results:
            return "detection"
        elif "system_level_correct" in eval_results or "fault_type_correct" in eval_results:
            return "analysis"
        elif "TTM" in eval_results:
            return "mitigation"
        elif "TTL" in eval_results:
            return "localization"
        
        # Fallback to task description keywords
        task_desc_lower = task_desc.lower()
        if "detect" in task_desc_lower:
            return "detection"
        elif "locali" in task_desc_lower:
            return "localization"
        elif "analysis" in task_desc_lower or "root cause" in task_desc_lower:
            return "analysis"
        elif "mitigat" in task_desc_lower or "fix" in task_desc_lower:
            return "mitigation"
        
        return "detection"  # Default
    
    def _format_task_info(self, task_info: Dict[str, Any]) -> str:
        """Format task info for prompt."""
        return json.dumps(task_info, indent=2, ensure_ascii=False)
    
    def _format_commands(self, commands: List[str]) -> str:
        """Format commands for prompt."""
        return "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(commands))
    
    def _select_fault_dimensions(self, num: int) -> List[str]:
        """Select fault dimensions for generation."""
        dimensions = self.config.fault_dimensions.copy()
        random.shuffle(dimensions)
        return dimensions[:num]
    
    async def generate_single(
        self,
        seed_data: Dict[str, Any],
        fault_dimension: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single candidate scenario.
        
        Args:
            seed_data: Parsed seed data
            fault_dimension: The fault dimension to focus on
            
        Returns:
            Generated scenario or None if failed
        """
        # Format prompt
        system_prompt = self._prompts["system_prompt"]
        generation_template = Template(self._prompts["generation_prompt"])
        
        user_prompt = generation_template.substitute(
            task_info=self._format_task_info(seed_data["task_info"]),
            commands=self._format_commands(seed_data["commands"]),
            fault_dimension=fault_dimension,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await self._call_llm(messages)
            
            # Parse JSON from response
            # Try to extract JSON from markdown code blocks if present (4 backticks have priority)
            if "````json" in response:
                json_start = response.find("````json") + 8
                json_end = response.find("````", json_start)
                response = response[json_start:json_end].strip()
            elif "````" in response:
                json_start = response.find("````") + 4
                json_end = response.find("````", json_start)
                response = response[json_start:json_end].strip()
            elif "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            scenario = json.loads(response)
            scenario["fault_dimension"] = fault_dimension
            scenario["source_seed"] = seed_data.get("task_info", {}).get("problem_id", "unknown")
            
            return scenario
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Failed to generate scenario: {e}")
            return None
    
    async def generate_batch(
        self,
        seed_data: Dict[str, Any],
        num_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple candidate scenarios in batch.
        
        This method generates N different scenarios focusing on different
        fault dimensions, implementing the multi-path generation strategy.
        
        Args:
            seed_data: Parsed seed data
            num_candidates: Number of candidates to generate (default: config value)
            
        Returns:
            List of generated scenarios
        """
        num = num_candidates or self.config.num_candidates
        
        # Select diverse fault dimensions
        dimensions = self._select_fault_dimensions(num)
        
        # Pad with random dimensions if we don't have enough
        while len(dimensions) < num:
            dimensions.append(random.choice(self.config.fault_dimensions))
        
        # Generate candidates in parallel
        tasks = [
            self.generate_single(seed_data, dim)
            for dim in dimensions[:num]
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        candidates = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Candidate {i} generation failed: {result}")
            elif result is not None:
                result["candidate_id"] = i + 1
                candidates.append(result)
        
        logger.info(f"Generated {len(candidates)}/{num} candidates successfully")
        return candidates
    
    async def generate_batch_single_call(
        self,
        seed_data: Dict[str, Any],
        num_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple candidates in a single LLM call.
        
        This is more efficient but may produce less diverse results.
        
        Args:
            seed_data: Parsed seed data
            num_candidates: Number of candidates to generate
            
        Returns:
            List of generated scenarios
        """
        num = num_candidates or self.config.num_candidates
        dimensions = self._select_fault_dimensions(num)
        
        # Format batch generation prompt
        system_prompt = self._prompts["system_prompt"]
        batch_template = Template(self._prompts["batch_generation_prompt"])
        
        user_prompt = batch_template.substitute(
            task_info=self._format_task_info(seed_data["task_info"]),
            commands=self._format_commands(seed_data["commands"]),
            num_candidates=num,
            fault_dimensions="\n".join(f"  - {dim}" for dim in dimensions),
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await self._call_llm(messages)
            
            # Parse JSON array from response (4 backticks have priority)
            if "````json" in response:
                json_start = response.find("````json") + 8
                json_end = response.find("````", json_start)
                response = response[json_start:json_end].strip()
            elif "````" in response:
                json_start = response.find("````") + 4
                json_end = response.find("````", json_start)
                response = response[json_start:json_end].strip()
            elif "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            candidates = json.loads(response)
            
            # Ensure it's a list
            if isinstance(candidates, dict):
                candidates = [candidates]
            
            # Add metadata
            source_id = seed_data.get("task_info", {}).get("problem_id", "unknown")
            for i, candidate in enumerate(candidates):
                candidate["candidate_id"] = i + 1
                candidate["source_seed"] = source_id
            
            return candidates
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batch LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Failed to generate batch: {e}")
            return []
    
    def format_output(
        self,
        candidate: Dict[str, Any],
        original_seed: Dict[str, Any],
        task_type: str,
    ) -> Dict[str, Any]:
        """
        Format a candidate into the output structure matching gt_sft_command format.
        
        Args:
            candidate: Generated candidate scenario
            original_seed: Original seed data for reference
            task_type: Task type (detection/localization/analysis/mitigation)
            
        Returns:
            Formatted output matching gt_sft_command structure
        """
        # Build task_info from candidate
        candidate_task_info = candidate.get("task_info", {})
        original_task_info = original_seed.get("task_info", {})
        
        # Generate new problem_id
        fault_dim = candidate.get("fault_dimension", "unknown")
        source_id = candidate.get("source_seed", "unknown")
        candidate_id = candidate.get("candidate_id", 1)
        new_problem_id = f"gen_{fault_dim}_{source_id}_{candidate_id}"
        
        # Build new task_info
        new_task_info = {
            "session_id": f"evolver-{new_problem_id}",
            "problem_id": new_problem_id,
            "status": "initialized",
            "task_description": candidate_task_info.get(
                "task_description",
                original_task_info.get("task_description", "")
            ),
            "instructions": original_task_info.get("instructions", ""),
            "available_actions": original_task_info.get("available_actions", {}),
            "checkpoints": ["initial"],
            # Additional metadata
            "fault_summary": candidate_task_info.get("fault_summary", ""),
            "root_cause": candidate.get("root_cause", ""),
            "expected_findings": candidate.get("expected_findings", ""),
            "difficulty_level": candidate.get("difficulty_level", "medium"),
            "source_seed": source_id,
            "fault_dimension": fault_dim,
        }
        
        # Get system_state_summary from candidate (critical for SFT training)
        system_state_summary = candidate.get("system_state_summary", "")
        
        # Get commands from candidate
        commands = candidate.get("commands", [])
        
        # Build evaluation_results placeholder based on task type
        if task_type == "detection":
            eval_results = {
                "Detection Accuracy": "Pending",
                "TTD": 0.0,
                "steps": 0,
                "in_tokens": 0,
                "out_tokens": 0,
                "total_duration_seconds": 0.0,
            }
        elif task_type == "analysis":
            eval_results = {
                "system_level_correct": None,
                "fault_type_correct": None,
                "success": None,
                "TTA": 0.0,
                "steps": 0,
                "in_tokens": 0,
                "out_tokens": 0,
                "total_duration_seconds": 0.0,
            }
        elif task_type == "mitigation":
            eval_results = {
                "TTM": 0.0,
                "steps": 0,
                "in_tokens": 0,
                "out_tokens": 0,
                "success": None,
                "total_duration_seconds": 0.0,
            }
        else:  # localization
            eval_results = {
                "TTL": 0.0,
                "steps": 0,
                "in_tokens": 0,
                "out_tokens": 0,
                "success": None,
                "total_duration_seconds": 0.0,
            }
        
        # Return format matching gt_sft_command structure
        result = {
            "problem_id": new_problem_id,
            "system_state_summary": system_state_summary,
            "command_list": commands,
            # Also include task_info and evaluation_results for compatibility
            "task_info": new_task_info,
            "evaluation_results": eval_results,
        }
        
        return result
    
    async def process_seed_file(
        self,
        seed_path: str,
        output_dir: Optional[str] = None,
        num_candidates: Optional[int] = None,
        use_batch_call: bool = False,
    ) -> List[str]:
        """
        Process a single seed file and generate candidates.
        
        Args:
            seed_path: Path to the seed JSON file
            output_dir: Output directory (default: config value)
            num_candidates: Number of candidates to generate
            use_batch_call: Whether to use single batch call
            
        Returns:
            List of output file paths
        """
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Parse seed data
        seed_data = self._parse_seed_data(seed_path)
        task_type = self._detect_task_type(seed_data)
        
        logger.info(f"Processing seed: {seed_path}, task_type: {task_type}")
        
        # Generate candidates
        if use_batch_call:
            candidates = await self.generate_batch_single_call(seed_data, num_candidates)
        else:
            candidates = await self.generate_batch(seed_data, num_candidates)
        
        if not candidates:
            logger.warning(f"No candidates generated for {seed_path}")
            return []
        
        # Format and save outputs
        output_files = []
        seed_name = Path(seed_path).stem
        
        for candidate in candidates:
            formatted = self.format_output(candidate, seed_data, task_type)
            
            # Generate output filename
            candidate_id = candidate.get("candidate_id", 1)
            fault_dim = candidate.get("fault_dimension", "unknown")
            output_name = f"{seed_name}_gen_{fault_dim}_{candidate_id}.json"
            output_path = Path(output_dir) / output_name
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)
            
            output_files.append(str(output_path))
            logger.info(f"Saved: {output_path}")
        
        return output_files
    
    async def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        num_candidates: Optional[int] = None,
        use_batch_call: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Process all seed files in a directory.
        
        Args:
            input_dir: Input directory containing seed JSON files
            output_dir: Output directory
            num_candidates: Number of candidates per seed
            use_batch_call: Whether to use single batch call
            
        Returns:
            Dict mapping seed file to list of output files
        """
        input_path = Path(input_dir)
        seed_files = list(input_path.glob("*.json"))
        
        logger.info(f"Found {len(seed_files)} seed files in {input_dir}")
        
        results = {}
        for seed_file in seed_files:
            try:
                output_files = await self.process_seed_file(
                    str(seed_file),
                    output_dir,
                    num_candidates,
                    use_batch_call,
                )
                results[str(seed_file)] = output_files
            except Exception as e:
                logger.error(f"Failed to process {seed_file}: {e}")
                results[str(seed_file)] = []
        
        return results
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Synchronous wrapper for convenience
class LLMEvolverSync:
    """Synchronous wrapper for LLMEvolver."""
    
    def __init__(self, config: Optional[EvolverConfig] = None):
        self._evolver = LLMEvolver(config)
    
    def generate_batch(
        self,
        seed_data: Dict[str, Any],
        num_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate batch synchronously."""
        return asyncio.run(self._evolver.generate_batch(seed_data, num_candidates))
    
    def process_seed_file(
        self,
        seed_path: str,
        output_dir: Optional[str] = None,
        num_candidates: Optional[int] = None,
    ) -> List[str]:
        """Process seed file synchronously."""
        return asyncio.run(self._evolver.process_seed_file(
            seed_path, output_dir, num_candidates
        ))
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        num_candidates: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Process directory synchronously."""
        return asyncio.run(self._evolver.process_directory(
            input_dir, output_dir, num_candidates
        ))
