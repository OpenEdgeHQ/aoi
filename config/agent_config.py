from dataclasses import dataclass
from typing import Optional


@dataclass
class RollbackConfig:
    """回滚工具配置"""
    validate_rollback: bool = True
    retry_wait_time: int = 5
    clear_replicaset: bool = True
    clear_rs_wait_time: int = 10
    output_dir: str = "./rollback_output"
    namespace: Optional[str] = None


@dataclass
class AgentSystemConfig:
    """智能体系统配置"""
    api_key: str
    model_name: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.1
    max_steps: int = 20
    memory_provider: str = "aworld"
    rollback_config: Optional[RollbackConfig] = None

    def __post_init__(self):
        if self.rollback_config is None:
            self.rollback_config = RollbackConfig()