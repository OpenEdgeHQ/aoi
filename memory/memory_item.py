# memory/memory_item.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import uuid


class AgentType(Enum):
    """智能体类型"""
    OBSERVER = "observer"
    PROBE = "probe"
    EXECUTOR = "executor"
    COMPRESSOR = "compressor"


class MemoryType(Enum):
    """记忆类型"""
    RAW_CONTEXT = "raw_context"
    COMPRESSED_CONTEXT = "compressed_context"
    SUB_TASK = "sub_task"
    BASELINE_CONTEXT = "baseline_context"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"  # 待执行
    EXECUTING = "executing"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 跳过


@dataclass
class BaseMemoryItem:
    """基础记忆项"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self):
        """更新时间戳"""
        self.updated_at = datetime.now()


@dataclass
class SubTaskItem(BaseMemoryItem):
    """子任务项 - 增强版本"""
    task_name: str = ""
    task_description: str = ""
    task_objective: str = ""
    target_agent: Optional[AgentType] = None
    priority: int = 5
    task_context: Dict[str, Any] = field(default_factory=dict)

    # 执行相关
    status: TaskStatus = TaskStatus.PENDING
    execution_rounds: int = 0
    max_rounds: int = 5
    executor_id: Optional[str] = None

    # 新增字段
    iteration_number: int = 0  # 对应的迭代轮次
    is_submit_task: bool = False  # 是否是提交任务
    parent_task_id: Optional[str] = None  # 父任务ID（用于跟踪）
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他子任务ID

    # 执行结果
    result: Optional[str] = None
    error_message: Optional[str] = None
    completion_time: Optional[datetime] = None

    def __post_init__(self):
        """初始化后处理"""
        self.memory_type = MemoryType.SUB_TASK

    def start_execution(self, executor_id: str):
        """开始执行任务"""
        self.status = TaskStatus.EXECUTING
        self.executor_id = executor_id
        self.update()

    def complete_execution(self, success: bool = True, result: str = ""):
        """完成任务执行"""
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.result = result
        self.completion_time = datetime.now()
        self.update()

    def mark_failed(self, error_message: str = ""):
        """标记任务失败"""
        self.status = TaskStatus.FAILED
        self.error_message = error_message
        self.completion_time = datetime.now()
        self.update()

    def skip_task(self, reason: str = ""):
        """跳过任务"""
        self.status = TaskStatus.SKIPPED
        self.result = f"Skipped: {reason}"
        self.completion_time = datetime.now()
        self.update()

    def is_executable(self) -> bool:
        """
        检查任务是否可执行

        Returns:
            bool: 如果任务状态为PENDING则返回True，否则False

        Note:
            当前实现采用简单的线性任务队列，由Observer智能体控制执行顺序，
            因此不需要复杂的依赖检查机制。
        """
        return self.status == TaskStatus.PENDING

    def get_execution_info(self) -> Dict[str, Any]:
        """获取执行信息"""
        return {
            "task_id": self.id,
            "task_name": self.task_name,
            "objective": self.task_objective,
            "target_agent": self.target_agent.value if self.target_agent else None,
            "iteration": self.iteration_number,
            "status": self.status.value,
            "is_submit": self.is_submit_task,
            "execution_rounds": self.execution_rounds,
            "max_rounds": self.max_rounds
        }


@dataclass
class RawContextItem(BaseMemoryItem):
    """原始上下文项"""
    source_agent: AgentType = None
    source_agent_id: str = ""
    round_number: int = 0
    raw_output: Any = None
    command: Optional[str] = None
    execution_time: float = 0.0
    success: bool = True
    error_trace: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        self.memory_type = MemoryType.RAW_CONTEXT


@dataclass
class CompressedContextItem(BaseMemoryItem):
    """压缩后的上下文项"""
    source_items: List[str] = field(default_factory=list)
    compression_ratio: float = 0.0
    original_size: int = 0
    compressed_size: int = 0

    # 智能摘要
    summary: str = ""
    key_findings: List[Dict[str, Any]] = field(default_factory=list)
    anomaly_indicators: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # 语义信息
    semantic_tags: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

    # 压缩器信息
    compression_model: str = ""
    compression_prompt: str = ""

    def __post_init__(self):
        """初始化后处理"""
        self.memory_type = MemoryType.COMPRESSED_CONTEXT


@dataclass
class BaselineContextItem(BaseMemoryItem):
    """基线上下文项 - 存储前两个iteration的系统概览"""
    session_id: str = ""
    iteration_numbers: List[int] = field(default_factory=list)  # 包含的迭代轮次
    baseline_content: str = ""  # 基线内容（前两个iter的成功探测结果）
    commands_included: List[str] = field(default_factory=list)  # 包含的命令列表
    
    def __post_init__(self):
        """初始化后处理"""
        self.memory_type = MemoryType.BASELINE_CONTEXT