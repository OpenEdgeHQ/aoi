# Data Loader for GRPO Training
"""
Data loading utilities for GRPO training on AOI scenarios.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass

# Optional torch import
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = object  # Fallback base class

logger = logging.getLogger(__name__)


@dataclass
class SeedScenario:
    """A seed scenario loaded from JSON file."""
    
    file_path: str
    problem_id: str
    task_type: str  # detection, localization, analysis, mitigation
    task_description: str
    instructions: str
    available_actions: Dict[str, str]
    commands: List[str]
    service_name: str
    namespace: str
    evaluation_results: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "problem_id": self.problem_id,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "instructions": self.instructions,
            "available_actions": self.available_actions,
            "commands": self.commands,
            "service_name": self.service_name,
            "namespace": self.namespace,
            "evaluation_results": self.evaluation_results,
        }
    
    def get_prompt_context(self) -> str:
        """Get formatted context for prompt generation."""
        actions_str = "\n".join([
            f"  - {name}: {desc[:100]}..." if len(desc) > 100 else f"  - {name}: {desc}"
            for name, desc in self.available_actions.items()
        ])
        
        commands_str = "\n".join([f"  {i+1}. {cmd}" for i, cmd in enumerate(self.commands)])
        
        return f"""## Service Information
- Service Name: {self.service_name}
- Namespace: {self.namespace}
- Task Type: {self.task_type}

## Task Description
{self.task_description}

## Available Actions
{actions_str}

## Command Sequence (Ground Truth)
{commands_str}

## Evaluation Results
{json.dumps(self.evaluation_results, indent=2)}
"""


def detect_task_type(data: Dict[str, Any]) -> str:
    """Detect task type from seed data."""
    problem_id = data.get("task_info", {}).get("problem_id", "")
    eval_results = data.get("evaluation_results", {})
    task_desc = data.get("task_info", {}).get("task_description", "").lower()
    
    # Check problem_id first
    if "-detection-" in problem_id:
        return "detection"
    elif "-localization-" in problem_id:
        return "localization"
    elif "-analysis-" in problem_id:
        return "analysis"
    elif "-mitigation-" in problem_id:
        return "mitigation"
    
    # Fallback to evaluation results
    if "Detection Accuracy" in eval_results:
        return "detection"
    elif "system_level_correct" in eval_results:
        return "analysis"
    elif "TTM" in eval_results:
        return "mitigation"
    elif "TTL" in eval_results:
        return "localization"
    
    # Fallback to task description
    if "detect" in task_desc:
        return "detection"
    elif "locali" in task_desc:
        return "localization"
    elif "analysis" in task_desc or "root cause" in task_desc:
        return "analysis"
    elif "mitigat" in task_desc or "fix" in task_desc:
        return "mitigation"
    
    return "detection"


def extract_service_info(task_description: str) -> Tuple[str, str]:
    """Extract service name and namespace from task description."""
    service_name = "Unknown Service"
    namespace = "default"
    
    # Look for service name
    if "Service Name:" in task_description:
        lines = task_description.split("\n")
        for line in lines:
            if "Service Name:" in line:
                service_name = line.split("Service Name:")[-1].strip()
            elif "Namespace:" in line:
                namespace = line.split("Namespace:")[-1].strip()
    
    return service_name, namespace


def load_seed_from_file(file_path: str) -> Optional[SeedScenario]:
    """
    Load a single seed scenario from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        SeedScenario or None if loading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        task_info = data.get("task_info", {})
        task_description = task_info.get("task_description", "")
        service_name, namespace = extract_service_info(task_description)
        
        return SeedScenario(
            file_path=file_path,
            problem_id=task_info.get("problem_id", "unknown"),
            task_type=detect_task_type(data),
            task_description=task_description,
            instructions=task_info.get("instructions", ""),
            available_actions=task_info.get("available_actions", {}),
            commands=data.get("commands", []),
            service_name=service_name,
            namespace=namespace,
            evaluation_results=data.get("evaluation_results", {}),
        )
    except Exception as e:
        logger.error(f"Failed to load seed from {file_path}: {e}")
        return None


def load_seeds_from_directory(
    directory: str,
    max_seeds: Optional[int] = None,
    shuffle: bool = True,
    filter_successful: bool = True,
) -> List[SeedScenario]:
    """
    Load seed scenarios from a directory.
    
    Args:
        directory: Path to directory containing JSON files
        max_seeds: Maximum number of seeds to load
        shuffle: Whether to shuffle the seeds
        filter_successful: Only load successful/correct scenarios
        
    Returns:
        List of SeedScenario objects
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return []
    
    json_files = list(dir_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {directory}")
    
    if shuffle:
        random.shuffle(json_files)
    
    seeds = []
    for file_path in json_files:
        seed = load_seed_from_file(str(file_path))
        if seed is None:
            continue
        
        # Filter by success if requested
        if filter_successful:
            eval_results = seed.evaluation_results
            is_successful = False
            
            if "Detection Accuracy" in eval_results:
                is_successful = eval_results["Detection Accuracy"] == "Correct"
            elif "success" in eval_results:
                is_successful = eval_results["success"] is True
            elif "system_level_correct" in eval_results:
                is_successful = eval_results.get("system_level_correct", False) and \
                               eval_results.get("fault_type_correct", False)
            
            if not is_successful:
                logger.debug(f"Skipping unsuccessful seed: {seed.problem_id}")
                continue
        
        seeds.append(seed)
        
        if max_seeds and len(seeds) >= max_seeds:
            break
    
    logger.info(f"Loaded {len(seeds)} seeds")
    return seeds


class SeedDataset(Dataset):
    """PyTorch Dataset for seed scenarios."""
    
    def __init__(
        self,
        seeds: List[SeedScenario],
        tokenizer: Any = None,
        max_length: int = 2048,
    ):
        """
        Initialize the dataset.
        
        Args:
            seeds: List of SeedScenario objects
            tokenizer: Tokenizer for encoding (optional)
            max_length: Maximum sequence length
        """
        self.seeds = seeds
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.seeds)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seed = self.seeds[idx]
        
        item = {
            "seed_idx": idx,
            "seed": seed.to_dict(),
            "prompt_context": seed.get_prompt_context(),
        }
        
        # Tokenize if tokenizer is provided
        if self.tokenizer:
            encoded = self.tokenizer(
                item["prompt_context"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)
        
        return item


class GRPODataLoader:
    """
    Data loader optimized for GRPO training.
    
    Yields batches of seeds, where each seed will generate `group_size` candidates.
    """
    
    def __init__(
        self,
        seed_dir: str,
        batch_size: int = 2,
        group_size: int = 4,
        max_seeds: Optional[int] = None,
        shuffle: bool = True,
        filter_successful: bool = True,
        num_workers: int = 0,
    ):
        """
        Initialize the data loader.
        
        Args:
            seed_dir: Directory containing seed JSON files
            batch_size: Number of seeds per batch
            group_size: Number of candidates to generate per seed
            max_seeds: Maximum number of seeds to use
            shuffle: Whether to shuffle seeds each epoch
            filter_successful: Only use successful scenarios
            num_workers: Number of data loading workers
        """
        self.seed_dir = seed_dir
        self.batch_size = batch_size
        self.group_size = group_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Load seeds
        self.seeds = load_seeds_from_directory(
            seed_dir,
            max_seeds=max_seeds,
            shuffle=shuffle,
            filter_successful=filter_successful,
        )
        
        if not self.seeds:
            raise ValueError(f"No valid seeds found in {seed_dir}")
        
        logger.info(f"GRPODataLoader initialized with {len(self.seeds)} seeds")
        logger.info(f"Batch size: {batch_size}, Group size: {group_size}")
        logger.info(f"Effective samples per batch: {batch_size * group_size}")
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (len(self.seeds) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[SeedScenario]]:
        """Iterate over batches of seeds."""
        if self.shuffle:
            indices = list(range(len(self.seeds)))
            random.shuffle(indices)
            seeds = [self.seeds[i] for i in indices]
        else:
            seeds = self.seeds
        
        for i in range(0, len(seeds), self.batch_size):
            batch = seeds[i:i + self.batch_size]
            yield batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        task_types = {}
        for seed in self.seeds:
            task_types[seed.task_type] = task_types.get(seed.task_type, 0) + 1
        
        return {
            "total_seeds": len(self.seeds),
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "num_batches": len(self),
            "task_type_distribution": task_types,
        }


def create_grpo_prompt(seed: SeedScenario, fault_dimension: str) -> str:
    """
    Create a prompt for GRPO generation.
    
    Input fields from seed: problem_id, task_description, commands
    Output fields expected: problem_id, task_description, system_state_summary, commands
    
    Args:
        seed: The seed scenario
        fault_dimension: The fault dimension to focus on
        
    Returns:
        Formatted prompt string
    """
    # Get the actual number of commands in seed
    seed_commands_count = len(seed.commands)
    
    # Show ALL commands for reference
    commands_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(seed.commands))
    
    prompt = f"""You are an expert DevOps engineer. Create a NEW fault scenario based on the reference.

## Reference Scenario

**Problem ID**: {seed.problem_id}

**Task Description**:
{seed.task_description}

**Commands** ({seed_commands_count} total):
{commands_str}

## Your Task
Create a DIFFERENT {fault_dimension} fault scenario based on this reference.

Apply the generation methodology:
- Vary along independent dimensions (fault type, affected layer, failure mode)
- Consider alternative root causes for similar symptoms
- Explore the configuration space

## OUTPUT FORMAT (CRITICAL - MUST FOLLOW EXACTLY)

Output ONLY a JSON object inside a ````json code block (4 backticks):

````json
{{
  "problem_id": "your-fault-type-{seed.task_type}-1",
  "task_description": "Detailed description: service name, namespace, architecture, supported operations, and task objective for YOUR new fault scenario",
  "system_state_summary": "Comprehensive cluster state: 1) Root cause and symptoms, 2) Affected resources with realistic names (pods, services, nodes), 3) Error messages and log snippets, 4) Current resource status",
  "commands": [
    "exec_shell(\\"kubectl get pods -n {seed.namespace}\\")",
    "exec_shell(\\"kubectl describe pod pod-name -n {seed.namespace}\\")",
    "... diagnosis commands ...",
    "exec_shell(\\"kubectl apply/patch/scale ...\\")",
    "exec_shell(\\"kubectl get pods -n {seed.namespace}\\")"
  ]
}}
````

RULES:
- Output NOTHING before or after the JSON code block
- NO thinking, NO explanation, NO comments
- Exactly 4 fields: problem_id, task_description, system_state_summary, commands
- commands should be 15-40 items (optimal for the problem complexity)
- commands MUST include resolution commands (apply, patch, scale, delete, rollout)"""
    
    return prompt


def create_repair_prompt(
    seed: SeedScenario,
    failure_reason: str = "Task did not complete successfully - missing resolution or verification phase",
    system_state_summary: str = "",
) -> str:
    """
    Create a prompt for REPAIRING a failed task attempt.
    
    Unlike create_grpo_prompt which generates NEW scenarios, this function
    creates prompts that REPAIR the existing failed scenario by fixing
    the command sequence while keeping the same problem context.
    
    Args:
        seed: The failed seed scenario to repair
        failure_reason: Description of why the task failed
        system_state_summary: System state summary (optional, can be empty)
        
    Returns:
        Formatted repair prompt string
    """
    # Get the actual number of commands in seed
    seed_commands_count = len(seed.commands)
    
    # Show ALL commands for reference
    commands_str = "\n".join(f"  {i+1}. {cmd}" for i, cmd in enumerate(seed.commands))
    
    # Analyze the commands to provide hints about what's missing
    has_resolution = any(
        any(action in cmd.lower() for action in ['apply', 'patch', 'scale', 'delete', 'rollout', 'restart'])
        for cmd in seed.commands
    )
    has_verification = len(seed.commands) > 0 and any(
        'get pods' in cmd.lower() or 'get pod' in cmd.lower()
        for cmd in seed.commands[-3:]  # Check last 3 commands
    )
    
    missing_phases = []
    if not has_resolution:
        missing_phases.append("Resolution (no apply/patch/scale/delete commands found)")
    if not has_verification:
        missing_phases.append("Verification (no final status check found)")
    
    missing_phases_str = ", ".join(missing_phases) if missing_phases else "Unknown - analyze commands to determine"
    
    # Use provided system_state_summary or generate a placeholder
    if not system_state_summary:
        system_state_summary = f"[To be determined from diagnosis] Service: {seed.service_name}, Namespace: {seed.namespace}"
    
    prompt = f"""You are an expert DevOps engineer. Your task is to REPAIR a failed task attempt.

## FAILED TASK TO REPAIR

**Problem ID**: {seed.problem_id}

**Task Description**:
{seed.task_description}

**System State Summary**:
{system_state_summary}

**Failed Command Sequence** ({seed_commands_count} commands):
{commands_str}

**Failure Reason**:
{failure_reason}

**Analysis - Missing Phases**:
{missing_phases_str}

## YOUR TASK: REPAIR THIS COMMAND SEQUENCE

Analyze why this command sequence failed and generate a REPAIRED version.

**Repair Guidelines:**
1. KEEP the same problem_id ("{seed.problem_id}")
2. KEEP the same task_description (or make minimal clarifications)
3. PRESERVE valid commands from the original sequence
4. FIX missing, incorrect, or incomplete parts of the command sequence
5. ENSURE the repaired sequence includes ALL 4 phases:
   - Discovery: Initial resource inspection (get pods, services, deployments)
   - Diagnosis: Deep investigation (describe, logs, events, configs)
   - Resolution: FIX commands (apply, patch, scale, delete, rollout)
   - Verification: Confirm fix worked (get pods, describe, logs)

**DO NOT:**
- Generate a completely new/different scenario
- Change the problem context or task objective
- Remove valid diagnostic commands unnecessarily

## OUTPUT FORMAT (CRITICAL - MUST FOLLOW EXACTLY)

Output ONLY a JSON object inside a ````json code block (4 backticks):

````json
{{
  "problem_id": "{seed.problem_id}",
  "task_description": "[KEEP ORIGINAL or make minimal clarification]",
  "system_state_summary": "1) Root Cause and Symptoms: [description]. 2) Affected Resources: [list]. 3) Error Messages and Logs: [messages]. 4) Cluster Resource Status: [status].",
  "commands": [
    "// === DISCOVERY PHASE ===",
    "exec_shell(\\"kubectl get pods -n {seed.namespace}\\")",
    "exec_shell(\\"kubectl get events -n {seed.namespace} --sort-by='.lastTimestamp'\\")",
    "// === DIAGNOSIS PHASE ===",
    "exec_shell(\\"kubectl describe pod problematic-pod -n {seed.namespace}\\")",
    "exec_shell(\\"kubectl logs problematic-pod -n {seed.namespace}\\")",
    "// ... more diagnosis commands ...",
    "// === RESOLUTION PHASE ===",
    "exec_shell(\\"kubectl patch/apply/scale ...\\")",
    "// === VERIFICATION PHASE ===",
    "exec_shell(\\"kubectl get pods -n {seed.namespace}\\")",
    "exec_shell(\\"kubectl describe pod repaired-pod -n {seed.namespace}\\")"
  ]
}}
````

RULES:
- Output NOTHING before or after the JSON code block
- NO thinking, NO explanation, NO comments
- Keep original problem context - ONLY fix the commands
- Commands MUST be complete with all 4 phases (15-40 commands total)
- Commands MUST include resolution commands (apply, patch, scale, delete, rollout)"""
    
    return prompt


def collate_seeds(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of items from dataset
        
    Returns:
        Collated batch dictionary
    """
    collated = {
        "seed_idx": [item["seed_idx"] for item in batch],
        "seeds": [item["seed"] for item in batch],
        "prompt_contexts": [item["prompt_context"] for item in batch],
    }
    
    # Stack tensors if present and torch is available
    if TORCH_AVAILABLE and "input_ids" in batch[0]:
        collated["input_ids"] = torch.stack([item["input_ids"] for item in batch])
        collated["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    
    return collated

