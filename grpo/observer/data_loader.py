# Observer Data Loader for GRPO Training
"""
Data loading utilities for Observer GRPO training.
Handles:
1. Loading training data from JSON files
2. Building Training Model prompts (compressed_context)
3. Building Reward Model inputs (execution_history)
4. Task type balanced sampling
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ObserverSample:
    """A single training sample for Observer GRPO."""
    
    # Source file info
    file_path: str
    problem_id: str
    task_type: str  # detection, localization, analysis, mitigation
    
    # Task info
    task_description: str
    instructions: str
    available_actions: Dict[str, str]
    
    # Current iteration info
    current_iter: int  # n (1 to m-1)
    total_iters: int   # m
    expected_action: str  # probe, executor, submit
    
    # Steps data
    steps: List[Dict[str, Any]]
    
    # Precomputed prompts
    training_prompt: str = ""  # For Training Model
    execution_history: str = ""  # For Reward Model
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "problem_id": self.problem_id,
            "task_type": self.task_type,
            "current_iter": self.current_iter,
            "total_iters": self.total_iters,
            "expected_action": self.expected_action,
        }


def detect_task_type(problem_id: str) -> str:
    """Detect task type from problem_id."""
    if "-detection-" in problem_id:
        return "detection"
    elif "-localization-" in problem_id:
        return "localization"
    elif "-analysis-" in problem_id:
        return "analysis"
    elif "-mitigation-" in problem_id:
        return "mitigation"
    return "detection"


def extract_action_from_task(current_task: str) -> str:
    """
    Extract action type from current_task field.
    Example: "List Pods in Namespace (probe)" -> "probe"
    """
    match = re.search(r'\((\w+)\)\s*$', current_task)
    if match:
        return match.group(1)
    return "probe"


def get_task_type_info(task_type: str) -> str:
    """Get task type specific information."""
    task_type_info = {
        "detection": """**Task Type: DETECTION**
- Goal: Identify IF there is an issue and its general nature
- Submission: Usually submit("Yes") or submit("No") or describe the issue
- Focus: Symptom discovery and initial classification
- You need to find evidence of problems, not fix them""",
        
        "localization": """**Task Type: LOCALIZATION**
- Goal: Find the EXACT component containing the ROOT CAUSE
- Submission: Submit the specific component name (e.g., submit(["user-service"]))
- Focus: Distinguish ROOT CAUSE from SYMPTOMS
- Remember: Submit the SOURCE of the problem, not the VICTIM showing errors""",
        
        "analysis": """**Task Type: ANALYSIS**
- Goal: Classify the root cause into system level and fault type
- Submission: Usually submit("Application") or submit("Virtualization") etc.
- Focus: Understanding WHY the issue occurred and its category
- Classify based on where the DEFECT is, not where symptoms appear""",
        
        "mitigation": """**Task Type: MITIGATION**
- Goal: FIX the root cause and verify resolution
- Submission: Submit confirmation of fix (e.g., submit())
- Focus: Apply correct fix to ROOT CAUSE, then verify
- You MUST use executor agent to make changes"""
    }
    return task_type_info.get(task_type, task_type_info["detection"])


def get_iteration_warning(remaining_iterations: int) -> str:
    """Get iteration warning message."""
    if remaining_iterations <= 1:
        return "⚠️ CRITICAL: This is the LAST iteration! Current task should be submission!"
    elif remaining_iterations == 2:
        return "⚠️ WARNING: Only 2 iterations left!"
    else:
        return f"ℹ️ {remaining_iterations} iterations remaining."


def get_executor_reminder(task_type: str) -> str:
    """Get executor reminder based on task type."""
    if task_type in ["detection", "localization", "analysis"]:
        return "NO executor needed (investigation only)."
    elif task_type == "mitigation":
        return "USE executor for repairs."
    return ""


def get_analysis_guidance(current_iteration: int) -> str:
    """Get analysis guidance based on iteration."""
    if current_iteration == 1:
        return """**First Iteration - Initial Assessment**:
1. **Understand the task type**: The task type has been automatically determined from the problem ID
2. **Assess what you know**: What do you currently understand about the system?
3. **Decide next action**: What information or action do you need first?"""
    else:
        return """**Ongoing Iteration - Progress Check**:
1. **Review findings**: What have you learned? What gaps remain?
2. **Check progress**: Are you closer to achieving the task objective?
3. **Decide next action**: Continue investigating, start fixing, or submit?"""


def build_compressed_context(steps: List[Dict], current_iter: int) -> str:
    """
    Build compressed context for Training Model input.
    
    n = 1: No history
    n = 2: iter 1 command + result
    n > 2: iter 1 to n-2 summaries + iter n-1 command + result
    """
    if current_iter == 1:
        return "No previous iterations yet. This is the first iteration."
    
    if current_iter == 2:
        # n = 2: show iter 1 command + result
        step = steps[0]
        return f"""### Previous Iteration (iter 1)
**Command**: {step.get('command', 'N/A')}
**Result**:
{step.get('result', 'N/A')}"""
    
    # n > 2: summaries for iter 1 to n-2 + command/result for iter n-1
    lines = ["### Historical Summaries (iter 1 to {})".format(current_iter - 2)]
    
    for i in range(current_iter - 2):
        step = steps[i]
        summary = step.get('summary', 'N/A')
        lines.append(f"- **Iter {i + 1}**: {summary}")
    
    # Previous iteration (n-1)
    prev_step = steps[current_iter - 2]
    lines.append("")
    lines.append(f"### Previous Iteration (iter {current_iter - 1})")
    lines.append(f"**Command**: {prev_step.get('command', 'N/A')}")
    lines.append(f"**Result**:")
    lines.append(prev_step.get('result', 'N/A'))
    
    return "\n".join(lines)


def build_execution_history(steps: List[Dict]) -> str:
    """
    Build full execution history for Reward Model input.
    Contains all iterations (1 to m) with command + summary only (no result).
    """
    lines = []
    total = len(steps)
    
    for i, step in enumerate(steps):
        iter_num = i + 1
        action = extract_action_from_task(step.get('current_task', ''))
        command = step.get('command', 'N/A')
        summary = step.get('summary', 'N/A')
        
        # Mark final answer
        is_final = iter_num == total
        final_marker = " [FINAL ANSWER]" if is_final else ""
        
        lines.append(f"### Iteration {iter_num} ({action}){final_marker}")
        lines.append(f"**Command**: {command}")
        lines.append(f"**Summary**: {summary}")
        lines.append("")
    
    return "\n".join(lines)


def format_available_actions(actions: Dict[str, str]) -> str:
    """Format available actions for prompt."""
    lines = []
    for name, desc in actions.items():
        # Truncate long descriptions
        if len(desc) > 200:
            desc = desc[:200] + "..."
        lines.append(f"- **{name}**: {desc}")
    return "\n".join(lines)


def format_subtask_queue(steps: List[Dict], current_iter: int) -> str:
    """Format subtask queue status."""
    lines = [
        f"Total Tasks: {len(steps)}",
        f"Current Task: {current_iter}/{len(steps)}",
        "",
        "Task Queue:"
    ]
    
    for i, step in enumerate(steps):
        iter_num = i + 1
        task_name = step.get('current_task', f'Task {iter_num}')
        action = extract_action_from_task(task_name)
        
        if iter_num < current_iter:
            status = "✅"
        elif iter_num == current_iter:
            status = "▶️"
        else:
            status = "⏸"
        
        current_marker = " 👈 CURRENT" if iter_num == current_iter else ""
        lines.append(f"{iter_num}. {status} {task_name}{current_marker}")
    
    return "\n".join(lines)


def format_current_subtask(step: Dict, iter_num: int) -> str:
    """Format current subtask information."""
    task_name = step.get('current_task', f'Task {iter_num}')
    action = extract_action_from_task(task_name)
    
    return f"""Iteration: {iter_num}
Name: {task_name}
Target Agent: {action}
Status: executing"""


class ObserverDataLoader:
    """
    Data loader for Observer GRPO training.
    Handles balanced sampling across task types.
    """
    
    def __init__(
        self,
        data_dir: str,
        prompts_path: str,
        batch_size: int = 2,
        group_size: int = 4,
        shuffle: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.prompts_path = prompts_path
        self.batch_size = batch_size
        self.group_size = group_size
        self.shuffle = shuffle
        
        # Load prompt templates
        self._load_prompts()
        
        # Load all data files
        self._load_data_files()
        
        # Compute task type weights for balanced sampling
        self._compute_sample_weights()
        
        logger.info(f"ObserverDataLoader initialized with {len(self.data_files)} files")
        logger.info(f"Task distribution: {self.task_counts}")
    
    def _load_prompts(self):
        """Load prompt templates from YAML."""
        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        
        self.system_prompt_template = prompts.get('system_prompt', '')
        self.user_prompt_template = prompts.get('user_prompt', '')
    
    def _load_data_files(self):
        """Load all JSON data files and organize by task type."""
        self.data_files = list(self.data_dir.glob("*.json"))
        
        # Organize by task type
        self.files_by_type = {
            "detection": [],
            "localization": [],
            "analysis": [],
            "mitigation": [],
        }
        
        for file_path in self.data_files:
            task_type = detect_task_type(file_path.stem)
            self.files_by_type[task_type].append(file_path)
        
        # Count by type
        self.task_counts = {k: len(v) for k, v in self.files_by_type.items()}
    
    def _compute_sample_weights(self):
        """Compute sampling weights for balanced task type distribution."""
        total = sum(self.task_counts.values())
        if total == 0:
            self.sample_weights = {k: 0.25 for k in self.task_counts}
            return
        
        # Inverse frequency weighting
        weights = {}
        for task_type, count in self.task_counts.items():
            if count > 0:
                weights[task_type] = 1.0 / count
            else:
                weights[task_type] = 0.0
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.sample_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.sample_weights = {k: 0.25 for k in weights}
    
    def _sample_file(self) -> Path:
        """Sample a file using balanced task type weights."""
        # First select task type
        task_types = list(self.sample_weights.keys())
        weights = [self.sample_weights[t] for t in task_types]
        
        # Filter out empty task types
        valid_types = [(t, w) for t, w in zip(task_types, weights) if self.files_by_type[t]]
        if not valid_types:
            raise ValueError("No valid data files found")
        
        task_types, weights = zip(*valid_types)
        total = sum(weights)
        weights = [w / total for w in weights]
        
        selected_type = random.choices(task_types, weights=weights, k=1)[0]
        
        # Then select file from that type
        return random.choice(self.files_by_type[selected_type])
    
    def _sample_iter(self, total_iters: int) -> int:
        """Sample an iteration (1 to m-1)."""
        # n ranges from 1 to m-1 (we don't train on the final submit step itself)
        return random.randint(1, total_iters - 1)
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single JSON file with validation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {"steps": [], "_invalid": True}
        
        # Validate required fields
        steps = data.get("steps", [])
        if not steps or len(steps) < 2:
            logger.warning(f"Skipping {file_path}: empty or insufficient steps ({len(steps)})")
            return {"steps": [], "_invalid": True}
        
        if not data.get("problem_id"):
            logger.warning(f"Skipping {file_path}: missing problem_id")
            return {"steps": [], "_invalid": True}
        
        return data
    
    def _build_training_prompt(
        self,
        data: Dict[str, Any],
        current_iter: int,
    ) -> str:
        """Build the full training prompt for the model."""
        task_type = detect_task_type(data.get('problem_id', ''))
        steps = data.get('steps', [])
        total_iters = len(steps)
        remaining = total_iters - current_iter
        
        # Get current step
        current_step = steps[current_iter - 1] if current_iter <= len(steps) else {}
        
        # Build compressed context
        compressed_context = build_compressed_context(steps, current_iter)
        
        # Prepare system prompt variables
        system_vars = {
            "task_description": data.get('task_description', ''),
            "task_type_info": get_task_type_info(task_type),
            "api_instruction": data.get('instructions', ''),
            "available_actions": format_available_actions(data.get('available_actions', {})),
            "submit_format": "Use submit() API to submit your final answer.",
        }
        
        # Prepare user prompt variables
        user_vars = {
            "iteration_number": current_iter,
            "max_iterations": total_iters,
            "remaining_iterations": remaining,
            "iteration_warning": get_iteration_warning(remaining),
            "task_type": task_type.capitalize(),
            "executor_reminder": get_executor_reminder(task_type),
            "compressed_context": compressed_context,
            "formatted_current_subtask": format_current_subtask(current_step, current_iter),
            "subtask_queue_status": format_subtask_queue(steps, current_iter),
            "analysis_guidance": get_analysis_guidance(current_iter),
        }
        
        # Build prompts using template substitution
        from string import Template
        
        system_prompt = Template(self.system_prompt_template).safe_substitute(system_vars)
        user_prompt = Template(self.user_prompt_template).safe_substitute(user_vars)
        
        # Combine for chat format
        # For Qwen3, add /no_think suffix
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}\n\n/no_think"
        
        return full_prompt
    
    def create_sample(self, file_path: Path, current_iter: int) -> ObserverSample:
        """Create a single training sample."""
        data = self._load_file(file_path)
        
        problem_id = data.get('problem_id', '')
        task_type = detect_task_type(problem_id)
        steps = data.get('steps', [])
        total_iters = len(steps)
        
        # Get expected action from current step
        current_step = steps[current_iter - 1] if current_iter <= len(steps) else {}
        expected_action = extract_action_from_task(current_step.get('current_task', ''))
        
        # Build prompts
        training_prompt = self._build_training_prompt(data, current_iter)
        execution_history = build_execution_history(steps)
        
        return ObserverSample(
            file_path=str(file_path),
            problem_id=problem_id,
            task_type=task_type,
            task_description=data.get('task_description', ''),
            instructions=data.get('instructions', ''),
            available_actions=data.get('available_actions', {}),
            current_iter=current_iter,
            total_iters=total_iters,
            expected_action=expected_action,
            steps=steps,
            training_prompt=training_prompt,
            execution_history=execution_history,
        )
    
    def __len__(self) -> int:
        """Number of batches per epoch (approximate)."""
        # Each file can generate multiple samples (one per iter)
        total_samples = sum(
            len(self._load_file(f).get('steps', [])) - 1
            for f in self.data_files[:5]  # Sample estimate
        ) / 5 * len(self.data_files)
        return int(total_samples / self.batch_size)
    
    def __iter__(self) -> Iterator[List[ObserverSample]]:
        """Iterate over batches of samples."""
        # Generate samples for one epoch
        samples = []
        
        # Sample batch_size samples
        for _ in range(self.batch_size):
            file_path = self._sample_file()
            data = self._load_file(file_path)
            total_iters = len(data.get('steps', []))
            
            if total_iters < 2:
                continue
            
            current_iter = self._sample_iter(total_iters)
            sample = self.create_sample(file_path, current_iter)
            samples.append(sample)
        
        yield samples
    
    def get_batch(self) -> List[ObserverSample]:
        """Get a single batch of samples."""
        return next(iter(self))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total_files": len(self.data_files),
            "task_counts": self.task_counts,
            "sample_weights": self.sample_weights,
            "batch_size": self.batch_size,
            "group_size": self.group_size,
        }


def create_dataset_from_loader(
    data_loader: ObserverDataLoader,
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Create a HuggingFace-compatible dataset from the data loader.
    
    If num_samples is None, creates one sample per (file, iteration) combination,
    with balanced sampling across task types.
    
    Returns list of dicts with 'prompt' key for TRL GRPOTrainer.
    """
    samples = []
    
    if num_samples is None:
        # Create samples from all files with balanced task types
        # Calculate total iterations per task type
        task_type_samples = {k: [] for k in data_loader.files_by_type.keys()}
        
        for task_type, files in data_loader.files_by_type.items():
            for file_path in files:
                data = data_loader._load_file(file_path)
                
                # Skip invalid/empty files
                if data.get("_invalid"):
                    continue
                
                total_iters = len(data.get('steps', []))
                if total_iters < 2:
                    continue
                
                # Generate samples for each iteration (1 to m-1)
                for iter_n in range(1, total_iters):
                    sample = data_loader.create_sample(file_path, iter_n)
                    task_type_samples[task_type].append({
                        "prompt": sample.training_prompt,
                        "problem_id": sample.problem_id,
                        "task_type": sample.task_type,
                        "task_description": sample.task_description,
                        "current_iter": sample.current_iter,
                        "total_iters": sample.total_iters,
                        "expected_action": sample.expected_action,
                        "execution_history": sample.execution_history,
                        "steps": sample.steps,
                        "file_path": sample.file_path,
                    })
        
        # Balance across task types using oversampling
        max_samples = max(len(s) for s in task_type_samples.values()) if task_type_samples else 0
        
        for task_type, type_samples in task_type_samples.items():
            if not type_samples:
                continue
            
            # Oversample to match max
            if len(type_samples) < max_samples:
                # Repeat samples to reach max_samples
                multiplier = (max_samples // len(type_samples)) + 1
                type_samples = (type_samples * multiplier)[:max_samples]
            
            samples.extend(type_samples)
        
        # Shuffle
        random.shuffle(samples)
        
        logger.info(f"Created balanced dataset with {len(samples)} samples")
        for task_type, type_samples in task_type_samples.items():
            logger.info(f"  {task_type}: {len(type_samples)} -> {min(len(type_samples), max_samples) if type_samples else 0} (balanced)")
    else:
        # Original behavior: random sampling
        for _ in range(num_samples):
            batch = data_loader.get_batch()
            for sample in batch:
                samples.append({
                    "prompt": sample.training_prompt,
                    "problem_id": sample.problem_id,
                    "task_type": sample.task_type,
                    "task_description": sample.task_description,
                    "current_iter": sample.current_iter,
                    "total_iters": sample.total_iters,
                    "expected_action": sample.expected_action,
                    "execution_history": sample.execution_history,
                    "steps": sample.steps,
                    "file_path": sample.file_path,
                })
    
    return samples
