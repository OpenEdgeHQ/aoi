from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
import json
import logging

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig
from aworld.runner import Runners

from memory.memory_manager import MemoryManager
from memory.memory_item import AgentType
from utils.logger_config import AgentLogger


class BaseAgent(ABC):
    """基础智能体类 - 所有智能体的抽象基类"""

    def __init__(self,
                 name: str,
                 agent_type: AgentType,
                 llm_config: AgentConfig,
                 memory_manager: Optional[MemoryManager] = None,
                 max_iterations: int = 6):
        """
        初始化基础智能体

        Args:
            name: 智能体名称
            agent_type: 智能体类型
            llm_config: LLM配置
            memory_manager: 内存管理器
            max_iterations: 最大迭代次数
        """
        self.name = name
        self.agent_type = agent_type
        self.llm_config = llm_config
        self.memory_manager = memory_manager
        self.max_iterations = max_iterations

        # 执行状态
        self.current_iteration = 0
        self.session_id = None
        self.task_context = {}

        # Token使用统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

        # 日志（保留标准logger用于错误处理）
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # 使用AgentLogger进行日志输出（根据agent类型自动启用/禁用）
        self.agent_logger = AgentLogger(agent_type.value.upper())

        # 初始化LLM智能体
        self.llm_agent = self._initialize_llm_agent()

    @abstractmethod
    def _initialize_llm_agent(self) -> Agent:
        """初始化LLM智能体 - 子类必须实现"""
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """获取系统提示词 - 子类必须实现"""
        pass

    async def process(self,
                      task_instruction,  # 统一使用task_instruction
                      context: Dict[str, Any],
                      **kwargs) -> str:
        """
        处理任务的通用接口

        Args:
            task_instruction: 任务指令（来自Observer）
            context: 上下文信息
            **kwargs: 额外参数

        Returns:
            处理结果字符串
        """
        self.current_iteration += 1

        # 构建输入
        input_data = self._prepare_input(task_instruction, context, **kwargs)

        self.agent_logger.file_only(f"\n{'=' * 80}\n🤖 Agent: {self.name}\n{'=' * 80}")
        self.agent_logger.file_only(f"📝 Input Data:\n{input_data}\n{'=' * 80}")

        # self.agent_logger.info(f"\n{'=' * 80}\n🤖 Agent: {self.name}\n{'=' * 80}")
        # self.agent_logger.info(f"📝 Input Data:\n{input_data}\n{'=' * 80}")

        # 调用LLM获取决策
        decision = await self._get_llm_decision(input_data)

        self.agent_logger.file_only(f"\n{'=' * 80}\n🤖 Agent: {self.name}\n{'=' * 80}")
        self.agent_logger.file_only(f"📝 Decision:\n{decision}\n{'=' * 80}")
        # self.agent_logger.info(f"\n{'=' * 80}\n🤖 Agent: {self.name}\n{'=' * 80}")
        # self.agent_logger.info(f"📝 Decision:\n{decision}\n{'=' * 80}")

        # 处理决策结果
        output = self._process_decision(decision, context)

        return output

    @abstractmethod
    def _prepare_input(self,
                       task_instruction,  # 统一命名
                       context: Dict[str, Any],
                       **kwargs) -> str:
        """准备LLM输入 - 子类必须实现"""
        pass

    async def _get_llm_decision(self, input_data: str) -> Dict[str, Any]:
        """
        获取LLM决策

        Args:
            input_data: 输入数据

        Returns:
            LLM决策结果
        """
        try:
            # 调用LLM
            result = await Runners.run(
                input=input_data,
                agent=self.llm_agent
            )

            # 记录token使用情况（如果result包含usage信息）
            if hasattr(result, 'usage') and result.usage:
                usage = result.usage
                # Handle both dict and object formats
                if isinstance(usage, dict):
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                    total = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                else:
                    input_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
                    total = getattr(usage, 'total_tokens', 0) or (input_tokens + output_tokens)
                
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_tokens += total

            # 解析响应
            response = result.answer
            decision = self._parse_llm_response(response)

            return decision

        except Exception as e:
            self.logger.error(f"LLM decision error: {e}")

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应为结构化数据

        Args:
            response: LLM原始响应

        Returns:
            解析后的决策字典
        """
        try:
            # 尝试提取JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            return json.loads(json_str)

        except Exception as e:
            self.agent_logger.warning(f"Failed to parse LLM response: {e}")
            # 返回原始响应
            return {"raw_response": response}

    @abstractmethod
    def _process_decision(self,
                          decision: Dict[str, Any],
                          context: Dict[str, Any]) -> str:
        """处理决策结果 - 子类必须实现"""
        pass


    def reset(self):
        """重置智能体状态"""
        self.current_iteration = 0
        self.session_id = datetime.now().isoformat()
        self.task_context = {}
        # 不重置token统计，保留整个任务期间的累计值
        self.agent_logger.info(f"{self.name} reset completed")

    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "session_id": self.session_id,
            "active": self.current_iteration > 0
        }
    
    def get_token_usage(self) -> Dict[str, int]:
        """获取token使用统计"""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens
        }