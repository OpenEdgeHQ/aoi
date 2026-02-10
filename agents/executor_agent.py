from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json
import re

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig

from agents.base_agent import BaseAgent
from agents.probe_agent import ProbeAgent
from agents.file_reader_agent import FileReaderAgent
from memory.memory_manager import MemoryManager
from memory.memory_item import (
    AgentType, SubTaskItem, RawContextItem, BaselineContextItem, MemoryType
)
from prompts.loader import get_prompt_loader
from utils.text_utils import deduplicate_text


class ExecutorAgent(BaseAgent):
    """执行器智能体 - 负责修复故障"""

    def __init__(self,
                 llm_config: AgentConfig,
                 memory_manager: MemoryManager,
                 probe_agent: Optional[ProbeAgent] = None,
                 max_iterations: int = 5,
                 use_probe: bool = False,
                 task_description: str = "",
                 available_actions: Dict[str, str] = None,
                 api_instruction: str = ""):  # 改名：instructions -> api_instruction
        """
        初始化执行器智能体

        Args:
            llm_config: LLM配置
            memory_manager: 内存管理器
            probe_agent: 探测器智能体实例（可选）
            max_iterations: 最大执行轮次
            use_probe: 是否允许LLM自主决定使用探测器
            task_description: 任务描述
            available_actions: 可用的API动作
            api_instruction: API使用说明格式
        """
        self.task_description = task_description
        self.available_actions = available_actions or {}
        self.api_instruction = api_instruction or ""  # 改名
        self.use_probe = use_probe

        # 初始化或创建探测器
        if not probe_agent and use_probe:
            self.probe_agent = ProbeAgent(
                llm_config=llm_config,
                memory_manager=memory_manager,
                max_iterations=1,  # 单次探测
                task_description=task_description,
                available_actions=available_actions,
                api_instruction=api_instruction
            )
        else:
            self.probe_agent = probe_agent

        self.prompt_loader = get_prompt_loader()

        super().__init__(
            name="Executor Agent",
            agent_type=AgentType.EXECUTOR,
            llm_config=llm_config,
            memory_manager=memory_manager,
            max_iterations=max_iterations
        )

        # agent_logger 已在基类中初始化
        self.execution_results = []
        self.execution_round = 0
        self.successful_commands = []  # 成功的命令历史
        self.failed_commands = []  # 失败的命令历史
        self.failure_summaries = []  # 失败总结累积列表（新增）
        self.file_reader = FileReaderAgent(llm_config)  # 文件读取agent
        self.baseline_context = ""  # 从探测器获取的基线上下文

    def _initialize_llm_agent(self) -> Agent:
        """初始化LLM智能体"""
        return Agent(
            name=self.name,
            conf=self.llm_config,
            system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        """获取系统提示词 - 执行修复版本"""
        # 根据是否启用探测器，选择不同的probe_section
        if self.use_probe:
            probe_section = """
## Probe Usage (Enabled)
You can decide to probe the system for more information before executing repair:
- Set "use_probe": true and provide "probe_instruction" when you need more data
- Set "use_probe": false when you have enough information to proceed
- Probe is useful when: unclear error states, need to verify system status, complex failures
"""
        else:
            probe_section = """
## Probe Usage (Disabled)
Direct probe access is not available. You must proceed with the information provided.
Focus on making targeted fixes based on the diagnosis from the Observer.
"""

        return self.prompt_loader.get_prompt(
            agent_type="executor",
            prompt_type="system",
            probe_section=probe_section,
            available_actions=self._format_available_actions(),
            api_instruction=self.api_instruction
        )

    def _classify_result(self, result: str) -> bool:
        """
        分类执行结果是否成功
        Args:
            result: 执行结果字符串

        Returns:
            True if successful, False if error
        """
        # 转换为字符串（如果是dict）
        if isinstance(result, dict):
            result = json.dumps(result)

        # 检查是否包含错误标记
        error_patterns = [
            r'^Error:',  # 以Error:开头
            r'^[ERROR]'
            r'Error from server',  # Kubernetes服务器错误
            r'\(NotFound\)',  # Kubernetes NotFound错误
            r'\(Forbidden\)',  # Kubernetes Forbidden错误
            r'\(BadRequest\)',  # Kubernetes BadRequest错误
            r'\(Conflict\)',  # Kubernetes Conflict错误
            r'\(InternalError\)',  # Kubernetes InternalError
            r'\(Unauthorized\)',  # Kubernetes Unauthorized错误
            r'"error"\s*:\s*true',  # JSON中的error: true
            r'Error parsing response',  # 解析错误
            r'No API call found',  # API调用未找到
            r'command not found',  # 命令未找到
            r'permission denied',  # 权限拒绝
            r'no such file',  # 文件不存在
            r'cannot access',  # 无法访问
            r'failed to',  # 失败
            r'unable to',  # 无法
            r'not found',  # 未找到
            r'does not exist',  # 不存在
            r'already exists',  # 已存在（某些情况下是错误）
            r'connection refused',  # 连接被拒绝
            r'timeout',  # 超时
            r'timed out',  # 超时
        ]

        for pattern in error_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                return False

        return True

    def _prepare_input(self, task_instruction, context: Dict[str, Any], **kwargs) -> str:  # 改名
        """准备LLM输入"""
        execution_round = kwargs.get("execution_round", self.execution_round)
        successful_history = kwargs.get("successful_execution_history", self.successful_commands)
        failed_history = kwargs.get("failed_execution_history", self.failed_commands)
        probe_result = kwargs.get("probe_result", None)
        current_subtask = kwargs.get("current_subtask", None)
        executor_context = kwargs.get("executor_context", "")
        
        # 判断使用哪种prompt模板
        use_probe_prompt = kwargs.get("use_probe_prompt", False)
        use_error_analysis_prompt = kwargs.get("use_error_analysis_prompt", False)
        use_with_error_analysis_prompt = kwargs.get("use_with_error_analysis_prompt", False)
        failed_command = kwargs.get("failed_command", "")
        error_message = kwargs.get("error_message", "")
        error_analysis_report = kwargs.get("error_analysis_report", "")
        
        # 获取当前 iteration 编号
        current_iteration = current_subtask.iteration_number if current_subtask else 1

        # 构建probe相关的上下文（完整探测结果）
        probe_context = ""
        if probe_result and not use_probe_prompt:  # 常规场景显示在probe_context
            # 限制长度
            max_probe_ctx_length = 8000  # 约2000 tokens
            if len(probe_result) > max_probe_ctx_length:
                probe_context = (
                    probe_result[:max_probe_ctx_length] + 
                    f"\n\n... [Truncated: {len(probe_result) - max_probe_ctx_length} chars omitted] ..."
                )
            else:
                probe_context = probe_result
        
        # 将JSON格式的executor_context转换为Markdown格式
        executor_context_display = self._format_executor_context_to_markdown(executor_context)
        
        # 不截断executor_context，保留完整信息供执行器使用
        
        # 从executor_context中提取关键字段（用于简短提示）
        root_cause_component = "Unknown"
        root_cause_issue = "Unknown"
        namespace = "Unknown"
        
        if isinstance(executor_context, dict):
            root_cause = executor_context.get('root_cause', {})
            root_cause_component = root_cause.get('component', 'Unknown')
            root_cause_issue = root_cause.get('issue', 'Unknown')
            
            resources = executor_context.get('resources', {})
            namespace = resources.get('namespace', 'Unknown')

            fix_strategy = executor_context.get('fix_strategy', {})
            fix_method = fix_strategy.get('method', 'Unknown')
        
        # 准备参数
        params = {
            "task_instruction": task_instruction,
            "successful_execution_history": self._format_execution_history(successful_history, success=True, current_iteration=current_iteration),
            "failed_execution_history": self._format_execution_history(failed_history, success=False, current_iteration=current_iteration),
            "probe_context": probe_context,
            "executor_context": executor_context_display,
            "execution_round": execution_round,
            "max_iterations": self.max_iterations,
            # 额外的关键字段（用于简短提示）
            "root_cause_component": root_cause_component,
            "root_cause_issue": root_cause_issue,
            "namespace": namespace,
            "fix_method": fix_method
        }
        
        # 根据场景选择不同的prompt模板
        if use_error_analysis_prompt:
            # 错误分析模式：只分析错误
            params["failed_command"] = failed_command
            params["error_message"] = error_message
            prompt_type = "user_error_analysis"
        elif use_with_error_analysis_prompt:
            # 基于错误分析生成命令模式
            params["error_analysis_report"] = error_analysis_report
            prompt_type = "user_with_error_analysis"
        elif use_probe_prompt:
            # 使用探测后的专用prompt
            # 限制probe_result长度
            max_probe_length = 8000  # 约2000 tokens
            if probe_result and len(probe_result) > max_probe_length:
                probe_result_display = (
                    probe_result[:max_probe_length] + 
                    f"\n\n... [Truncated: {len(probe_result) - max_probe_length} chars omitted] ..."
                )
            else:
                probe_result_display = probe_result or ""
            params["probe_result"] = probe_result_display  # 探测结果放在专门的位置
            prompt_type = "user_with_probe"
        else:
            # 使用标准prompt
            prompt_type = "user"

        return self.prompt_loader.get_prompt(
            agent_type="executor",
            prompt_type=prompt_type,
            **params
        )

    def _format_available_actions(self) -> str:
        """格式化可用动作"""
        if not self.available_actions:
            return "No specific actions defined - use standard kubectl commands"

        formatted = []
        for action_name, action_desc in self.available_actions.items():
            # 保留完整的API文档
            formatted.append(f"**{action_name}**: {action_desc}")

        return "\n\n".join(formatted)

    def _format_failure_summaries(self) -> str:
        """
        格式化累积的失败总结
        
        Returns:
            格式化的失败总结字符串
        """
        if not self.failure_summaries:
            return "No previous failure summaries in this task period."
        
        formatted = []
        formatted.append(f"**Total Failures Analyzed**: {len(self.failure_summaries)}\n")
        
        for i, summary in enumerate(self.failure_summaries, 1):
            formatted.append(f"### Failure #{i} (Iter {summary['iteration']}, Round {summary['round']})")
            formatted.append(f"**Failed Command**: `{summary['failed_command']}`")
            formatted.append(f"**Error**: {summary['error']}")
            formatted.append(f"**Analysis**: {summary['analysis']}")
            formatted.append("")  # 空行分隔
        
        return "\n".join(formatted)
    
    def _format_executor_context_to_markdown(self, executor_context) -> str:
        """
        将JSON格式的executor_context转换为清晰的Markdown格式
        
        Args:
            executor_context: Observer提供的诊断信息，可能是dict或str
            
        Returns:
            str: 格式化的Markdown文本
        """
        # 如果是字符串，尝试解析为JSON
        if isinstance(executor_context, str):
            if not executor_context or executor_context.strip() == "":
                return "No diagnostic context provided"
            # 尝试解析JSON
            try:
                import json
                executor_context = json.loads(executor_context)
            except:
                # 如果不是JSON，直接返回原文本（兼容旧格式）
                return executor_context
        
        # 如果不是dict，返回空
        if not isinstance(executor_context, dict):
            return str(executor_context)
        
        md_parts = []
        
        # 1. 问题类型
        problem_type = executor_context.get('problem_type', 'Unknown')
        md_parts.append(f"### 🔍 Problem Type\n**{problem_type}**\n")
        
        # 2. 根因分析
        root_cause = executor_context.get('root_cause', {})
        if root_cause:
            md_parts.append("### 🎯 Root Cause (THIS IS WHAT YOU NEED TO FIX)\n")
            md_parts.append(f"- **Component**: `{root_cause.get('component', 'Unknown')}`")
            md_parts.append(f"- **Issue**: {root_cause.get('issue', 'Unknown')}")
            
            evidence = root_cause.get('evidence', [])
            if evidence:
                md_parts.append("\n**Evidence (Verbatim Errors)**:")
                for i, err in enumerate(evidence, 1):
                    md_parts.append(f"{i}. `{err}`")
            md_parts.append("")
        
        # 3. 症状（不要修复这些）
        symptoms = executor_context.get('symptoms', [])
        if symptoms:
            md_parts.append("### 📊 Symptoms (Effects - Don't Fix These)\n")
            for symptom in symptoms:
                comp = symptom.get('component', 'Unknown')
                status = symptom.get('status', 'Unknown')
                desc = symptom.get('description', '')
                md_parts.append(f"- **{comp}**: `{status}`")
                if desc:
                    md_parts.append(f"  - {desc}")
            md_parts.append("")
        
        # 4. 资源信息
        resources = executor_context.get('resources', {})
        if resources:
            md_parts.append("### 🏷️ Resources (Use Exact Names)\n")
            
            namespace = resources.get('namespace')
            if namespace:
                md_parts.append(f"- **Namespace**: `{namespace}` ⚠️ REQUIRED")
            
            for key in ['affected_services', 'affected_pods', 'affected_deployments']:
                values = resources.get(key, [])
                if values:
                    label = key.replace('_', ' ').replace('affected ', '').title()
                    items = ', '.join([f'`{v}`' for v in values])
                    md_parts.append(f"- **{label}**: {items}")
            
            config_details = resources.get('config_details', {})
            if config_details:
                md_parts.append("\n**Configuration**:")
                for k, v in config_details.items():
                    md_parts.append(f"- {k}: {v}")
            md_parts.append("")

        # 5. 修复策略（如果有）
        fix_strategy = executor_context.get('fix_strategy', {})
        if fix_strategy:
            md_parts.append("### 🔧 Fix Strategy\n")

            method = fix_strategy.get('method', '')
            if method:
                md_parts.append(f"**Method**: {method}")
            
            priority = fix_strategy.get('priority', '')
            if priority:
                md_parts.append(f"**Priority**: {priority}")

            commands = fix_strategy.get('commands', [])
            if commands:
                md_parts.append("\n**Commands**:")
                for i, cmd in enumerate(commands, 1):
                    md_parts.append(f"{i}. `{cmd}`")

            verification_steps = fix_strategy.get('verification_steps', [])
            if verification_steps:
                md_parts.append("\n**Verification Steps**:")
                for i, step in enumerate(verification_steps, 1):
                    md_parts.append(f"{i}. {step}")

            fallback_plan = fix_strategy.get('fallback_plan', '')
            if fallback_plan:
                md_parts.append(f"\n**Fallback Plan**: {fallback_plan}")
            
            md_parts.append("")
        
        return "\n".join(md_parts)

    def _format_execution_history(self, execution_history, success: bool = True, current_iteration: int = None) -> str:
        """
        格式化执行历史
        - 成功命令：总是显示命令+完整结果（执行结果通常简短）
        - 失败命令：只显示最后一个的命令+错误，其他只显示命令
        """
        if not execution_history:
            return f"No {'successful' if success else 'failed'} commands yet."

        history_type = "✓ SUCCESSFUL" if success else "✗ FAILED"

        # 根据成功/失败显示不同数量
        display_count = 10 if success else 5

        # 获取最近的命令
        recent_items = execution_history[-display_count:]
        
        formatted_commands = []
        
        for i, item in enumerate(recent_items):
            if isinstance(item, dict):
                command = item.get("command", "")
                result = item.get("result", "")
                item_iteration = item.get("iteration", 0)
                item_round = item.get("round", 0)
            else:
                # 兼容旧格式（字符串）
                command = item
                result = ""
                item_iteration = 0
                item_round = 0
            
            if success:
                # 成功命令：总是显示完整结果（执行结果通常简短）
                if result:
                    # 执行结果通常简短，不截断
                    formatted_commands.append(f"  - Iter {item_iteration}, Round {item_round}: {command}\n    ✅ Result: {result}")
                else:
                    formatted_commands.append(f"  - {command}")
            else:
                # 失败命令：只有最后一个显示错误，其他只显示命令
                if i == len(recent_items) - 1 and result:
                    error_preview = result[:300] + "..." if len(result) > 300 else result
                    formatted_commands.append(f"  - {command}\n    ❌ Error: {error_preview}")
                else:
                    formatted_commands.append(f"  - {command}")
        
        # 如果历史超过显示数量，添加提示
        header = ""
        if len(execution_history) > display_count:
            header = f"[{history_type}] Showing last {display_count} of {len(execution_history)} commands:\n"
        else:
            header = f"[{history_type}] {len(execution_history)} commands:\n"
        
        return header + "\n".join(formatted_commands)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        parsed = {
            "raw_response": response,
            "analysis": "",
            "use_probe": False,
            "probe_instruction": "",
            "executor_command": None,
            "next_action": "CONTINUE",
            # 错误分析模式字段
            "error_category": "",
            "root_cause": "",
            "improvement_direction": ""
        }

        try:
            # 尝试JSON解析
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())

                # 提取所有字段
                for key in ["analysis", "use_probe", "probe_instruction",
                            "executor_command", "next_action",
                            "error_category", "root_cause", "improvement_direction"]:
                    if key in json_data:
                        parsed[key] = json_data[key]

        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")

        # 如果没有找到命令，尝试其他方式提取
        if not parsed["executor_command"] and not parsed["use_probe"] and not parsed["error_category"]:
            command_patterns = [
                r'"executor_command"\s*:\s*"([^"]+)"',
                r'`([^`]+)`',  # 代码块中的命令
                r'kubectl\s+[^\n]+',  # kubectl命令
                r'exec_shell\(["\']([^"\']+)["\']\)',  # exec_shell格式
            ]

            for pattern in command_patterns:
                match = re.search(pattern, response)
                if match:
                    parsed["executor_command"] = match.group(1) if '(' in pattern else match.group(0)
                    break

        return parsed

    def _process_decision(self,
                          decision: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """处理决策结果"""
        self.execution_round += 1

        # 提取决策内容
        analysis = decision.get("analysis", "")
        use_probe = decision.get("use_probe", False)
        executor_command = decision.get("executor_command", "")
        next_action = decision.get("next_action", "CONTINUE")
        
        # 错误分析模式字段
        error_category = decision.get("error_category", "")
        root_cause = decision.get("root_cause", "")
        improvement_direction = decision.get("improvement_direction", "")

        # 显示分析
        if analysis:
            self.agent_logger.info(f"📝 Analysis: {analysis[:200]}...")

        # 如果是错误分析模式
        if error_category:
            self.agent_logger.info(f"🔍 Error Analysis Mode")
            self.agent_logger.info(f"  Category: {error_category}")
            self.agent_logger.info(f"  Root Cause: {root_cause[:150]}...")
            self.agent_logger.info(f"  Improvement: {improvement_direction[:150]}...")
            
            return {
                "round": self.execution_round,
                "mode": "error_analysis",
                "error_category": error_category,
                "root_cause": root_cause,
                "improvement_direction": improvement_direction,
                "next_action": next_action,
                "timestamp": datetime.now().isoformat()
            }

        if not use_probe and not executor_command:
            self.agent_logger.error("❌ No executor command found")
            return {
                "round": self.execution_round,
                "use_probe": False,
                "executor_command": "",
                "next_action": "COMPLETE",
                "error": "No executor command found",
                "timestamp": datetime.now().isoformat()
            }

        # 输出执行信息
        if use_probe:
            self.agent_logger.info(f"🔍 Round {self.execution_round}: Requesting probe first")
            self.agent_logger.info(f"📋 Probe instruction: {decision.get('probe_instruction', '')[:100]}...")
        else:
            self.agent_logger.info(f"⚡ Round {self.execution_round}: {executor_command[:80]}...")

        # 构建结果
        result = {
            "round": self.execution_round,
            "analysis": analysis,
            "use_probe": use_probe,
            "probe_instruction": decision.get("probe_instruction", "") if use_probe else "",
            "executor_command": executor_command if not use_probe else "",
            "next_action": next_action,
            "timestamp": datetime.now().isoformat()
        }

        self.execution_results.append(result)
        return result

    def _get_fallback_decision(self) -> Dict[str, Any]:
        """获取备用决策"""
        return {
            "use_probe": False,
            "executor_command": "",
            "next_action": "COMPLETE",
            "error": "Failed to get valid decision from LLM",
            "round": self.execution_round
        }

    async def executor_system(self,
                              task_instruction,
                              current_subtask: Optional[SubTaskItem] = None,
                              successful_execution_history: List[Dict] = None,
                              failed_execution_history: List[Dict] = None,
                              execution_round: int = None,
                              probe_result: Any = None,
                              executor_context: str = "",
                              use_probe_prompt: bool = False,
                              use_error_analysis_prompt: bool = False,
                              use_with_error_analysis_prompt: bool = False,
                              failed_command: str = "",
                              error_message: str = "",
                              error_analysis_report: str = "") -> Dict[str, Any]:
        """
        执行系统修复 - 主要接口

        Args:
            task_instruction: 子任务指示来自观察器
            current_subtask: 当前子任务
            successful_execution_history: 成功的执行历史
            failed_execution_history: 失败的执行历史
            execution_round: 当前执行轮次
            probe_result: 探测结果
            executor_context: 观察器提供的执行关键上下文
            use_probe_prompt: 是否使用探测后的prompt
            use_error_analysis_prompt: 是否使用错误分析模式
            use_with_error_analysis_prompt: 是否使用基于错误分析的命令生成模式
            failed_command: 失败的命令
            error_message: 错误信息
            error_analysis_report: 错误分析报告
        Returns:
            包含决策信息的字典
        """
        try:
            if execution_round is not None:
                self.execution_round = execution_round - 1  # 因为_process_decision会+1

            result = await self.process(
                task_instruction,
                context={},
                current_subtask=current_subtask,
                execution_round=execution_round or (self.execution_round + 1),
                successful_execution_history=successful_execution_history or [],
                failed_execution_history=failed_execution_history or [],
                probe_result=probe_result,
                executor_context=executor_context,
                use_probe_prompt=use_probe_prompt,
                use_error_analysis_prompt=use_error_analysis_prompt,
                use_with_error_analysis_prompt=use_with_error_analysis_prompt,
                failed_command=failed_command,
                error_message=error_message,
                error_analysis_report=error_analysis_report
            )

            if isinstance(result, dict):
                return result
            else:
                self.logger.error(f"Unexpected result type: {type(result)}")
                return self._get_fallback_decision()

        except Exception as e:
            self.logger.error(f"Error in executor_system: {e}")
            return self._get_fallback_decision()

    async def executor_run(self,
                           task_instruction,
                           execute_action: Callable,
                           current_subtask: Optional[SubTaskItem] = None,
                           session_id: Optional[str] = None,
                           executor_context: str = "") -> Dict[str, Any]:
        """
        运行完整执行流程（带重试机制）

        Args:
            task_instruction: 子任务指示来自观察器
            execute_action: 执行命令的函数
            current_subtask: 当前子任务
            session_id: 会话ID
            executor_context: 执行器上下文（来自Observer）

        Returns:
            执行结果汇总
        """
        # 重置状态
        self.reset()
        if session_id:
            self.session_id = session_id
        
        # 从 current_subtask 获取 iteration
        current_iteration = current_subtask.iteration_number if current_subtask else 1
        
        # 从memory加载baseline_context（如果还没有）
        if not self.baseline_context and self.memory_manager and session_id:
            baseline_items = self.memory_manager.query_items(
                agent_type=AgentType.PROBE,  # baseline是由probe创建的
                memory_type=MemoryType.BASELINE_CONTEXT,
                filters={"session_id": lambda x: x == session_id if isinstance(x, str) else x.get("session_id") == session_id},
                limit=1,
                sort_by="created_at",
                descending=True
            )
            if baseline_items and isinstance(baseline_items[0], BaselineContextItem):
                self.baseline_context = baseline_items[0].baseline_content
                self.agent_logger.info(f"💾 Loaded baseline_context from memory (iterations {baseline_items[0].iteration_numbers})")

        # 初始化
        all_results = []
        probe_result = None
        retry_count = 0  # 当前轮次的重试次数
        max_retries_per_round = 3  # 每轮最多重试次数

        self.agent_logger.info(f"🚀 Starting executor (max {self.max_iterations} rounds)")

        round_num = 1
        while round_num <= self.max_iterations:
            try:
                # 获取执行决策
                decision = await self.executor_system(
                    task_instruction=task_instruction,
                    current_subtask=current_subtask,
                    successful_execution_history=self.successful_commands,
                    failed_execution_history=self.failed_commands,
                    execution_round=round_num,
                    probe_result=probe_result,
                    executor_context=executor_context
                )

                round_data = {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "next_action": decision.get("next_action", "CONTINUE")
                }

                # 检查是否需要探测
                if decision.get("use_probe", False) and self.use_probe and self.probe_agent:
                    # 执行探测 - Executor调用时只执行1次快速探测
                    probe_instruction = decision.get("probe_instruction", "Investigate system state")
                    self.agent_logger.info(f"🔎 Calling probe agent (single round)...")

                    # 创建临时的单次探测器（max_iterations=1）
                    from agents.probe_agent import ProbeAgent
                    single_probe = ProbeAgent(
                        llm_config=self.llm_config,
                        memory_manager=self.memory_manager,
                        max_iterations=1,  # 关键：Executor调用时只执行1次
                        task_description=self.task_description,
                        available_actions=self.available_actions,
                        api_instruction=self.api_instruction
                    )
                    
                    # 关键：继承主probe agent的所有重要上下文
                    # baseline_context包含前两个iteration的基础信息（命名空间、服务列表等）
                    if hasattr(self.probe_agent, 'baseline_context'):
                        single_probe.baseline_context = self.probe_agent.baseline_context
                        self.agent_logger.info(f"📋 Inherited baseline_context ({len(self.probe_agent.baseline_context)} chars)")
                    
                    # 继承命令历史，避免重复执行
                    if hasattr(self.probe_agent, 'successful_commands'):
                        single_probe.successful_commands = self.probe_agent.successful_commands.copy()
                    if hasattr(self.probe_agent, 'failed_commands'):
                        single_probe.failed_commands = self.probe_agent.failed_commands.copy()
                    
                    probe_res = await single_probe.probe_run(
                        task_instruction=probe_instruction,
                        execute_action=execute_action,
                        current_subtask=current_subtask,
                        session_id=session_id
                    )
                    
                    # 构建简洁的探测结果
                    probe_results_detail = probe_res.get("results", [])
                    
                    # 格式化探测结果
                    full_probe_result = []
                    full_probe_result.append(f"## Single Probe Result")
                    
                    if probe_results_detail:
                        result_item = probe_results_detail[0]  # 只有1个结果
                        if result_item.get("success", False):
                            cmd = result_item.get("command", "N/A")
                            output = result_item.get("result", "")
                            full_probe_result.append(f"**Command**: {cmd}")
                            full_probe_result.append(f"**Result**:\n```\n{output}\n```")
                        else:
                            error = result_item.get("error", "Unknown error")
                            full_probe_result.append(f"**Error**: {error}")
                    
                    probe_result = "\n".join(full_probe_result)
                    
                    # 记录日志
                    self.agent_logger.info(
                        f"📊 Probe completed: 1 operation, "
                        f"{'successful' if probe_results_detail and probe_results_detail[0].get('success') else 'failed'}"
                    )
                    
                    # 保存探测器返回的successful和failed命令列表
                    probe_successful = probe_res.get("successful_commands_list", [])
                    probe_failed = probe_res.get("failed_commands_list", [])
                    
                    # 合并到执行器的历史中（避免重复）
                    for cmd in probe_successful:
                        if cmd not in self.successful_commands:
                            self.successful_commands.append(cmd)
                    
                    for cmd in probe_failed:
                        if cmd not in self.failed_commands:
                            self.failed_commands.append(cmd)
                    
                    # 【关键】同步回主 probe_agent 的历史，确保跨 iteration 的命令历史一致性
                    if self.probe_agent:
                        for cmd in probe_successful:
                            if cmd not in self.probe_agent.successful_commands:
                                self.probe_agent.successful_commands.append(cmd)
                        
                        for cmd in probe_failed:
                            if cmd not in self.probe_agent.failed_commands:
                                self.probe_agent.failed_commands.append(cmd)
                        
                        self.agent_logger.info(
                            f"📝 Synced to probe_agent: {len(probe_successful)} successful, {len(probe_failed)} failed"
                        )
                    
                    self.agent_logger.info(
                        f"📝 Merged probe results to executor: {len(probe_successful)} successful, {len(probe_failed)} failed"
                    )

                    round_data["action_type"] = "probe"
                    round_data["command"] = probe_instruction
                    round_data["result"] = probe_result if probe_result else ""  # 不截断结果
                    round_data["success"] = True
                    round_data["full_probe_result"] = probe_result

                    # 添加到成功历史（探测本身）
                    probe_action_record = {
                        "command": f"[PROBE] {probe_instruction}", 
                        "result": probe_result,  # 不截断结果
                        "iteration": current_iteration,
                        "round": round_num
                    }
                    if probe_action_record not in self.successful_commands:
                        self.successful_commands.append(probe_action_record)

                    # 关键改进：立即利用探测结果生成修复命令，而不是继续到下一轮
                    self.agent_logger.info(f"🔄 Using probe results to generate repair command...")
                    
                    # 使用探测结果重新生成修复决策（使用专门的prompt模板）
                    retry_decision = await self.executor_system(
                        task_instruction=task_instruction,
                        current_subtask=current_subtask,
                        successful_execution_history=self.successful_commands,
                        failed_execution_history=self.failed_commands,
                        execution_round=round_num,
                        probe_result=probe_result,  # 传递探测结果
                        executor_context=executor_context,
                        use_probe_prompt=True  # 标记使用探测后的prompt
                    )
                    
                    # 使用新的决策替代原决策，更新 decision
                    if retry_decision.get("executor_command"):
                        decision["executor_command"] = retry_decision.get("executor_command")
                        self.agent_logger.info(f"✨ Generated repair command based on probe: {decision['executor_command'][:80]}...")
                        # 执行这个命令（下面的代码会处理）
                    else:
                        self.agent_logger.warning(f"⚠️ No repair command generated after probe")
                        round_num += 1
                        continue

                # 执行修复命令
                executor_command = decision.get("executor_command", "")
                if executor_command:
                    # 记录开始时间
                    start_time = datetime.now()
                    retry_used = False

                    # 执行命令
                    try:
                        exec_result = execute_action(executor_command)
                        
                        # 立即对结果进行去重处理
                        if exec_result and isinstance(exec_result, str) and len(exec_result) > 1000:
                            deduplicated_result, stats = deduplicate_text(exec_result)
                            if stats["reduction_ratio"] > 0.1:  # 只有去重效果超过10%才记录和使用
                                self.agent_logger.info(
                                    f"  📝 Deduplication: {stats['original_length']} → {stats['deduplicated_length']} chars "
                                    f"(reduced {stats['reduction_ratio']:.1%})"
                                )
                                exec_result = deduplicated_result
                        
                        # 检测并处理CSV文件（get_traces/get_metrics等）
                        if self.file_reader.should_read_files(executor_command):
                            enhanced_result, read_files = await self.file_reader.process_result(
                                command=executor_command,
                                result_text=exec_result,
                                task_instruction=task_instruction
                            )
                            if read_files:
                                exec_result = enhanced_result  # 使用增强后的结果
                        
                        is_success = self._classify_result(exec_result)

                        # 循环重试，直到成功或达到最大重试次数
                        while not is_success and retry_count < max_retries_per_round:
                            self.agent_logger.warning(
                                f"⚠️ Command failed, attempting retry {retry_count + 1}/{max_retries_per_round}")

                            # 记录失败的命令（包含错误信息）
                            self.failed_commands.append({
                                "command": executor_command, 
                                "result": exec_result,
                                "iteration": current_iteration,
                                "round": round_num
                            })

                            retry_count += 1
                            retry_used = True

                            # === 两阶段错误处理 ===
                            
                            # 阶段1：错误分析模式 - 只分析错误，不生成命令
                            self.agent_logger.info(f"🔍 Stage 1: Analyzing error...")
                            
                            error_analysis_decision = await self.executor_system(
                                task_instruction=task_instruction,
                                current_subtask=current_subtask,
                                successful_execution_history=self.successful_commands,
                                failed_execution_history=self.failed_commands,
                                execution_round=round_num,
                                probe_result=probe_result,
                                executor_context=executor_context,
                                use_error_analysis_prompt=True,  # 标记使用错误分析模式
                                failed_command=executor_command,
                                error_message=exec_result
                            )
                            
                            # 提取错误分析结果
                            error_category = error_analysis_decision.get("error_category", "unknown")
                            root_cause = error_analysis_decision.get("root_cause", "")
                            improvement_direction = error_analysis_decision.get("improvement_direction", "")
                            
                            # 构建错误分析报告
                            error_analysis_report = f"""
## Error Analysis Report

**Error Category**: {error_category}

**Root Cause**: 
{root_cause}

**Improvement Direction**:
{improvement_direction}
"""
                            
                            self.agent_logger.info(f"  ✅ Error analyzed: {error_category}")
                            self.agent_logger.info(f"  📋 Root cause: {root_cause[:100]}...")
                            
                            # 阶段2：基于错误分析生成新命令
                            self.agent_logger.info(f"🔄 Stage 2: Generating new command based on analysis...")
                            
                            retry_decision = await self.executor_system(
                                task_instruction=task_instruction,
                                current_subtask=current_subtask,
                                successful_execution_history=self.successful_commands,
                                failed_execution_history=self.failed_commands,
                                execution_round=round_num,
                                probe_result=probe_result,
                                executor_context=executor_context,
                                use_with_error_analysis_prompt=True,  # 标记使用基于错误分析的命令生成模式
                                error_analysis_report=error_analysis_report
                            )
                            
                            retry_command = retry_decision.get("executor_command", "")
                            if retry_command and retry_command != executor_command:  # 确保不是相同的命令
                                self.agent_logger.info(f"  ✨ New command generated: {retry_command[:80]}...")
                                exec_result = execute_action(retry_command)
                                
                                # 立即对结果进行去重处理
                                if exec_result and isinstance(exec_result, str) and len(exec_result) > 1000:
                                    deduplicated_result, stats = deduplicate_text(exec_result)
                                    if stats["reduction_ratio"] > 0.1:
                                        self.agent_logger.info(
                                            f"  📝 Deduplication: {stats['original_length']} → {stats['deduplicated_length']} chars "
                                            f"(reduced {stats['reduction_ratio']:.1%})"
                                        )
                                        exec_result = deduplicated_result
                                
                                is_success = self._classify_result(exec_result)
                                executor_command = retry_command  # 更新命令记录

                                if is_success:
                                    self.agent_logger.success(f"✅ Retry successful!")
                            else:
                                self.agent_logger.warning(f"⚠️ Could not generate alternative command, skipping retry")
                                break  # 无法生成新命令，停止重试

                        # 根据成功/失败分类存储
                        execution_time = (datetime.now() - start_time).total_seconds()
                        command_record = {
                            "round": round_num,
                            "action_type": "execute",
                            "command": executor_command,
                            "result": exec_result,
                            "retry_used": retry_used
                        }

                        if is_success:
                            # 记录成功的命令（包含结果）
                            self.successful_commands.append({
                                "command": executor_command, 
                                "result": exec_result, 
                                "iteration": current_iteration,
                                "round": round_num
                            })
                            self.agent_logger.info(f"✅ Command successful{' (after retry)' if retry_used else ''}")

                            # 只有成功的结果才保存到Memory
                            if self.memory_manager:
                                raw_item = RawContextItem(
                                    source_agent=self.agent_type,
                                    source_agent_id=self.session_id or "",
                                    round_number=round_num,
                                    raw_output=exec_result,
                                    command=executor_command,
                                    execution_time=execution_time,
                                    success=True,
                                    metadata={
                                        "session_id": self.session_id,
                                        "source_agent": self.agent_type.value,
                                        "iteration": current_iteration,
                                        "round_number": round_num,
                                        "command": executor_command,
                                        "result": exec_result,
                                        "retry_used": retry_used
                                    }
                                )
                                self.memory_manager.add_item(raw_item, self.agent_type)

                            # 成功后重置重试计数
                            retry_count = 0
                        else:
                            # 记录失败的命令（包含错误信息）
                            self.failed_commands.append({
                                "command": executor_command, 
                                "result": exec_result,
                                "iteration": current_iteration,
                                "round": round_num
                            })
                            self.agent_logger.warning(f"⚠️ Command failed after all retries")
                            # 提取错误信息
                            error_match = re.search(r'Error:\s*(.+?)(?:\n|$)', exec_result)
                            if error_match:
                                self.agent_logger.error(f"Error: {error_match.group(1)}")

                        # 显示结果预览
                        result_preview = exec_result[:200] + "..." if len(exec_result) > 200 else exec_result
                        self.agent_logger.info(f"📊 Result: {result_preview}")

                        round_data["action_type"] = "execute"
                        round_data["command"] = executor_command
                        round_data["result"] = exec_result if exec_result else ""  # 不截断结果
                        round_data["success"] = is_success
                        round_data["retry_used"] = retry_used

                    except Exception as e:
                        self.logger.error(f"Error executing command: {e}")

                        # 尝试重试
                        if retry_count < max_retries_per_round:
                            retry_count += 1
                            self.agent_logger.warning(
                                f"⚠️ Exception occurred, attempting retry {retry_count}/{max_retries_per_round}")

                            # 记录失败（包含错误信息）
                            self.failed_commands.append({
                                "command": executor_command, 
                                "result": f"Error: {str(e)}",
                                "iteration": current_iteration,
                                "round": round_num
                            })

                            # 跳到下一轮，让LLM生成新策略
                            round_num += 1
                            retry_count = 0
                            continue

                        exec_result = f"Error: {str(e)}"

                        # 记录失败（包含错误信息）
                        self.failed_commands.append({
                            "command": executor_command, 
                            "result": exec_result,
                            "iteration": current_iteration,
                            "round": round_num
                        })

                        round_data["action_type"] = "execute"
                        round_data["command"] = executor_command
                        round_data["result"] = exec_result  # 不截断结果
                        round_data["success"] = False

                        self.agent_logger.error(f"❌ Command failed: {str(e)}")

                all_results.append(round_data)

                self.agent_logger.info(
                    f"Round {round_num}: action={round_data.get('action_type')}, "
                    f"success={round_data.get('success', 'N/A')}, "
                    f"retry={round_data.get('retry_used', False)}, "
                    f"next={decision.get('next_action')}"
                )

                # 检查是否完成
                if decision.get("next_action") == "COMPLETE":
                    self.agent_logger.success(f"✨ Execution completed at round {round_num}")
                    break

                # 前进到下一轮
                round_num += 1
                retry_count = 0  # 重置重试计数

                # 检查是否达到最大轮次
                if round_num > self.max_iterations:
                    self.agent_logger.warning(f"⏱️ Reached maximum iterations ({self.max_iterations})")
                    break

            except Exception as e:
                self.logger.error(f"💥 Error in execution round {round_num}: {e}")
                error_info = {
                    "round": round_num,
                    "error": str(e),
                    "next_action": "ERROR",
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(error_info)
                break

        # 构建完整的执行历史字符串（只包含成功的）
        successful_history = "\n".join([
            f"Round {item['round']} [{item.get('action_type', 'execute').upper()}]: "
            f"{item.get('command', '')}\nResult: {item.get('result', '')}"  # 不截断结果
            for item in all_results if item.get('success', False)
        ])

        # 返回结果
        return {
            "total_rounds": len(all_results),
            "completed": True,
            "results": all_results,
            "execution_history": successful_history,
            "successful_commands": len(self.successful_commands),
            "failed_commands": len(self.failed_commands),
            "final_status": all_results[-1]["next_action"] if all_results else "NO_RESULTS",
            "session_id": self.session_id,
            "retries_used": sum(1 for r in all_results if r.get("retry_used", False))
        }

    def reset(self):
        """重置执行器状态"""
        super().reset()
        self.execution_results = []
        self.execution_round = 0
        # 不再重置命令历史，让它们在整个任务期间累积
        # self.successful_commands = []
        # self.failed_commands = []