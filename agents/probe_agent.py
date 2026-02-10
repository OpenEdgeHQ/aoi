from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import re

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig
from agents.base_agent import BaseAgent
from agents.file_reader_agent import FileReaderAgent
from memory.memory_manager import MemoryManager
from memory.memory_item import (
    AgentType, RawContextItem, SubTaskItem, BaselineContextItem, MemoryType
)
from prompts.loader import get_prompt_loader
from utils.text_utils import deduplicate_text


class ProbeAgent(BaseAgent):
    """探测器智能体 - 只读系统状态数据收集"""

    def __init__(self,
                 llm_config: AgentConfig,
                 memory_manager: MemoryManager,
                 max_iterations: int = 5,
                 task_description: str = "",
                 available_actions: Dict[str, str] = None,
                 api_instruction: str = ""):
        """
        初始化探测器智能体

        Args:
            llm_config: LLM配置
            memory_manager: 内存管理器
            max_iterations: 最大探测轮次
            task_description: 任务描述
            available_actions: 可用的API动作
            api_instruction: API使用说明
        """
        self.task_description = task_description
        self.available_actions = available_actions or {}
        self.api_instruction = api_instruction or ""  # 改名
        self.prompt_loader = get_prompt_loader()

        super().__init__(
            name="Probe Agent",
            agent_type=AgentType.PROBE,
            llm_config=llm_config,
            memory_manager=memory_manager,
            max_iterations=max_iterations
        )

        # agent_logger 已在基类中初始化
        self.probe_results = []
        self.probe_round = 0
        self.successful_commands = []  # 成功的命令历史
        self.failed_commands = []  # 失败的命令历史
        self.file_reader = FileReaderAgent(llm_config)  # 文件读取agent
        self.baseline_context = ""  # 前两个iter的基础信息（命名空间、服务列表等）

    def _initialize_llm_agent(self) -> Agent:
        """初始化LLM智能体"""
        return Agent(
            name=self.name,
            conf=self.llm_config,
            system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        # 现在需要传入available_actions和api_instruction
        return self.prompt_loader.get_prompt(
            agent_type="probe",
            prompt_type="system",
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
            r'^Error',  # 以Error开头
            r'Error from server',  # Kubernetes API 错误
            r'Syntax error',
            # r'^No \w+ found',
            # r'Error parsing response',  # 解析错误
            # r'No API call found',  # API调用未找到
            # r'command not found',  # 命令未找到
            # r'permission denied',  # 权限拒绝
            # r'no such file or directory',  # 文件不存在
            # r'cannot access',  # 无法访问
            # r'failed to',  # 失败
            # r'unable to'  # 无法
        ]

        for pattern in error_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                return False

        return True

    def _prepare_input(self, task_instruction, context: Dict[str, Any],
                       **kwargs) -> str:  # 改名：instruction -> task_instruction
        """准备LLM输入"""
        probe_round = kwargs.get("probe_round", self.probe_round)
        successful_history = kwargs.get("successful_probe_history", self.successful_commands)
        failed_history = kwargs.get("failed_probe_history", self.failed_commands)
        probe_context = kwargs.get("probe_context", "")
        current_subtask = kwargs.get("current_subtask", None)
        
        # 获取当前 iteration 编号
        current_iteration = current_subtask.iteration_number if current_subtask else 1

        # 处理 baseline_context 的条件显示（不再截断，提供完整信息）
        if self.baseline_context:
            # 直接使用完整的baseline_context，不进行截断
            # LLM应该能够处理完整的历史信息以做出准确决策
            baseline_display = self.baseline_context
        else:
            baseline_display = "No baseline yet. Collect: namespaces, services, pods."
        
        # 将JSON格式的probe_context转换为Markdown格式
        probe_context_display = self._format_probe_context_to_markdown(probe_context)
        
        # 不截断probe_context，保留完整信息供探测器使用
        
        # 从probe_context中提取关键字段（用于简短提示）
        investigation_phase = "surface_scan"
        investigation_type = "health_check"
        primary_targets = []
        suggested_commands = []
        
        if isinstance(probe_context, dict):
            investigation_phase = probe_context.get('investigation_phase', 'surface_scan')
            
            investigation_focus = probe_context.get('investigation_focus', {})
            investigation_type = investigation_focus.get('investigation_type', 'health_check')
            primary_targets = investigation_focus.get('primary_targets', [])
            # 回滚：不再读取 suggested_commands
        
        # 回滚：不再展示 suggested_commands
        
        params = {
            "task_instruction": task_instruction,  # 改名
            "successful_probe_history": self._format_probe_history(successful_history, success=True, current_round=probe_round),
            "failed_probe_history": self._format_probe_history(failed_history, success=False, current_round=probe_round),
            "baseline_context": baseline_display,  # 前两个iter的基础信息
            "probe_context": probe_context_display,
            "probe_round": probe_round,
            "max_iterations": self.max_iterations,
            # 额外的关键字段（用于简短提示）
            "investigation_phase": investigation_phase,
            "investigation_type": investigation_type,
            "primary_targets": ", ".join(primary_targets) if primary_targets else "Not specified"
        }

        return self.prompt_loader.get_prompt(
            agent_type="probe",
            prompt_type="user",
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

    def _format_probe_history(self, probe_history, success: bool = True, current_round: int = None) -> str:
        """
        格式化探测历史
        - 成功命令：显示所有成功命令（从round 1开始），只显示命令不显示结果
        - 失败命令：只显示最后一个的命令+错误，其他只显示命令
        """
        if not probe_history:
            return f"No {'successful' if success else 'failed'} commands yet."

        history_type = "✓ SUCCESSFUL" if success else "✗ FAILED"

        # 根据成功/失败显示不同数量的历史
        display_count = 10 if success else 5

        # 不再过滤，显示所有历史（从round 1开始）
        filtered_history = probe_history

        if not filtered_history:
            return f"No {'successful' if success else 'failed'} commands yet."

        # 获取最近的命令
        recent_items = filtered_history[-display_count:]
        
        formatted_commands = []
        
        for i, item in enumerate(recent_items):
            if isinstance(item, dict):
                command = item.get("command", "")
                result = item.get("result", "")
                item_round = item.get("round", 0)
            else:
                # 兼容旧格式（字符串）
                command = item
                result = ""
                item_round = 0
            
            if success:
                # 成功命令：只显示命令，不显示结果
                formatted_commands.append(f"  - Round {item_round}: {command}")
            else:
                # 失败命令：只有最后一个显示完整错误，其他只显示命令
                if i == len(recent_items) - 1 and result:
                    # 不截断，显示完整错误信息
                    formatted_commands.append(f"  - {command}\n    ❌ Error: {result}")
                else:
                    formatted_commands.append(f"  - {command}")

        # 如果历史超过显示数量，添加提示
        header = ""
        if len(filtered_history) > display_count:
            header = f"[{history_type}] Showing last {display_count} of {len(filtered_history)} commands:\n"
        else:
            header = f"[{history_type}] {len(filtered_history)} commands:\n"
        
        return header + "\n".join(formatted_commands)

    def _format_probe_context_to_markdown(self, probe_context) -> str:
        """
        将JSON格式的probe_context转换为清晰的Markdown格式
        
        Args:
            probe_context: Observer提供的调查指导信息，可能是dict或str
            
        Returns:
            str: 格式化的Markdown文本
        """
        # 如果是字符串，尝试解析为JSON
        if isinstance(probe_context, str):
            if not probe_context or probe_context.strip() == "":
                return "No investigation context provided"
            # 尝试解析JSON
            try:
                probe_context = json.loads(probe_context)
            except:
                # 如果不是JSON，直接返回原文本（兼容旧格式）
                return probe_context
        
        # 如果不是dict，返回空
        if not isinstance(probe_context, dict):
            return str(probe_context)
        
        md_parts = []
        
        # 1. 调查阶段
        investigation_phase = probe_context.get('investigation_phase', 'Not specified')
        phase_emoji = {
            'surface_scan': '🔍',
            'point_investigation': '🎯',
            'depth_analysis': '🔬',
            'verification': '✅'
        }
        emoji = phase_emoji.get(investigation_phase, '📋')
        md_parts.append(f"### {emoji} Investigation Phase\n**{investigation_phase.replace('_', ' ').title()}**\n")
        
        # 2. 调查焦点
        investigation_focus = probe_context.get('investigation_focus', {})
        if investigation_focus:
            md_parts.append("### 🎯 Investigation Focus\n")
            
            primary_targets = investigation_focus.get('primary_targets', [])
            if primary_targets:
                md_parts.append(f"**Primary Targets**: {', '.join([f'`{t}`' for t in primary_targets])}\n")
            
            investigation_type = investigation_focus.get('investigation_type', '')
            if investigation_type:
                md_parts.append(f"**Investigation Type**: {investigation_type}\n")
            
            specific_checks = investigation_focus.get('specific_checks', [])
            if specific_checks:
                md_parts.append("\n**Specific Checks**:")
                for check in specific_checks:
                    md_parts.append(f"- {check}")
                md_parts.append("")  # 空行
            
            # 回滚：不再展示 suggested_commands
        
        # 3. 比较分析要求
        comparison_requirements = probe_context.get('comparison_requirements', {})
        if comparison_requirements:
            need_comparison = comparison_requirements.get('need_comparison', False)
            if need_comparison:
                md_parts.append("### 📊 Comparison Requirements\n")
                
                comparison_targets = comparison_requirements.get('comparison_targets', [])
                if comparison_targets:
                    md_parts.append("**Comparison Targets**:")
                    for target in comparison_targets:
                        md_parts.append(f"- {target}")
                    md_parts.append("")
                
                outlier_detection = comparison_requirements.get('outlier_detection', '')
                if outlier_detection:
                    md_parts.append(f"**Outlier Detection**: {outlier_detection}\n")
        
        return "\n".join(md_parts)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应为结构化数据"""
        parsed = {
            "raw_response": response,
            "probe_command": None,
            "next_action": "CONTINUE",
            "reasoning": "",
            "focus_area": "",
            "analysis": ""
        }

        try:
            # 尝试JSON解析
            json_match = re.search(r'\{.*}', response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())

                # 提取所有字段
                for key in ["probe_command", "next_action", "reasoning",
                            "focus_area", "analysis"]:
                    if key in json_data:
                        parsed[key] = json_data[key]

        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")

        # 如果没有找到命令，尝试其他方式提取
        if not parsed["probe_command"]:
            # 查找可能的命令模式
            command_patterns = [
                r'"probe_command"\s*:\s*"([^"]+)"',
                r'`([^`]+)`',  # 代码块中的命令
                r'kubectl\s+[^\n]+',  # kubectl命令
                r'exec_shell\(["\']([^"\']+)["\']\)',  # exec_shell格式
            ]

            for pattern in command_patterns:
                match = re.search(pattern, response)
                if match:
                    parsed["probe_command"] = match.group(1) if '(' in pattern else match.group(0)
                    break

        return parsed

    def _process_decision(self,
                          decision: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """处理决策结果"""
        self.probe_round += 1

        # 获取探测命令和其他信息
        probe_command = decision.get("probe_command", "")
        next_action = decision.get("next_action", "CONTINUE")
        reasoning = decision.get("reasoning", "")
        focus_area = decision.get("focus_area", "")
        analysis = decision.get("analysis", "")

        if not probe_command:
            self.agent_logger.error("❌ No probe command found")
            return {
                "probe_command": "",
                "next_action": "COMPLETE",
                "error": "No probe command found",
                "round": self.probe_round
            }

        # 输出探测信息
        self.agent_logger.info(f"🔍 Round {self.probe_round}: {probe_command[:80]}...")
        if focus_area:
            self.agent_logger.info(f"🎯 Focus: {focus_area}")

        # 构建结果
        result = {
            "probe_command": probe_command,
            "next_action": next_action,
            "reasoning": reasoning,
            "focus_area": focus_area,
            "analysis": analysis,
            "round": self.probe_round,
            "timestamp": datetime.now().isoformat()
        }

        self.probe_results.append(result)
        return result

    def _get_fallback_decision(self) -> Dict[str, Any]:
        """获取备用决策"""
        return {
            "probe_command": "",
            "next_action": "COMPLETE",
            "error": "Failed to get valid decision from LLM",
            "round": self.probe_round
        }

    async def probe_system(self,
                           task_instruction,  # 改名：instruction -> task_instruction
                           current_subtask: Optional[SubTaskItem] = None,
                           successful_probe_history: List[Dict] = None,
                           failed_probe_history: List[Dict] = None,
                           probe_round: int = None,
                           probe_context: str = "") -> Dict[str, Any]:
        """
        执行系统探测 - 主要接口

        Args:
            task_instruction: 子任务指引来自观察器
            current_subtask: 当前子任务
            successful_probe_history: 成功的探测历史
            failed_probe_history: 失败的探测历史
            probe_round: 当前探测轮次
            probe_context: 探测器上下文（来自Observer）
        """
        try:
            if probe_round is not None:
                self.probe_round = probe_round - 1  # 因为_process_decision会+1

            result = await self.process(
                task_instruction,
                context={},
                current_subtask=current_subtask,
                probe_round=probe_round or (self.probe_round + 1),
                successful_probe_history=successful_probe_history or [],
                failed_probe_history=failed_probe_history or [],
                probe_context=probe_context,
            )

            if isinstance(result, dict):
                return result
            else:
                self.logger.error(f"Unexpected result type: {type(result)}")
                return self._get_fallback_decision()

        except Exception as e:
            self.logger.error(f"Error in probe_system: {e}")
            return self._get_fallback_decision()

    async def probe_run(self,
                        task_instruction: str,
                        execute_action: Callable[[str], str],
                        current_subtask: Optional[SubTaskItem] = None,
                        session_id: Optional[str] = None,
                        probe_context: str = "") -> Dict[str, Any]:
        """
        运行完整的多轮探测流程（带重试机制）
        Args:
            task_instruction: 子任务指引来自观察器
            execute_action: 执行命令的函数
            current_subtask: 当前子任务
            session_id: 会话ID
            probe_context: 探测器上下文（来自Observer）
        """
        # 重置状态
        self.reset()
        if session_id:
            self.session_id = session_id
        
        # 从 current_subtask 获取 iteration
        current_iteration = current_subtask.iteration_number if current_subtask else 1

        # 初始化
        all_results = []
        retry_count = 0  # 当前轮次的重试次数
        max_retries_per_round = 3  # 每轮最多重试次数

        self.agent_logger.info(f"🚀 Starting probe (max {self.max_iterations} rounds)")

        probe_round = 1
        while probe_round <= self.max_iterations:
            try:
                # 调用探测系统
                probe_result = await self.probe_system(
                    task_instruction=task_instruction,
                    current_subtask=current_subtask,
                    successful_probe_history=self.successful_commands,
                    failed_probe_history=self.failed_commands,
                    probe_round=probe_round,
                    probe_context=probe_context,
                )

                # 提取命令和下一步动作
                command = probe_result.get("probe_command", "")
                next_action = probe_result.get("next_action", "CONTINUE")
                focus_area = probe_result.get("focus_area", "")
                analysis = probe_result.get("analysis", "")

                # 执行命令
                exec_result = ""
                is_success = True
                retry_used = False

                if command:
                    try:
                        exec_result = execute_action(command)

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
                        if self.file_reader.should_read_files(command):
                            enhanced_result, read_files = await self.file_reader.process_result(
                                command=command,
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

                            # 记录失败的命令（包含错误信息，用于t-1显示）
                            self.failed_commands.append({"command": command, "result": exec_result})

                            retry_count += 1
                            retry_used = True

                            # 立即重新生成并执行探测命令
                            self.agent_logger.info(f"🔄 Retrying with adjusted approach...")

                            # 重新调用probe_system生成新命令（会考虑失败历史）
                            # 提供完整的错误信息，不进行截断，以便LLM能够准确分析失败原因
                            retry_result = await self.probe_system(
                                task_instruction=task_instruction + f"\n[RETRY HINT] Previous command '{command}' failed with: {exec_result}",
                                current_subtask=current_subtask,
                                successful_probe_history=self.successful_commands,
                                failed_probe_history=self.failed_commands,
                                probe_round=probe_round,
                                probe_context=probe_context,
                            )

                            retry_command = retry_result.get("probe_command", "")
                            if retry_command and retry_command != command:  # 确保不是相同的命令
                                exec_result = execute_action(retry_command)
                                is_success = self._classify_result(exec_result)
                                command = retry_command  # 更新命令记录

                                if is_success:
                                    self.agent_logger.success(f"✅ Retry successful!")
                            else:
                                self.agent_logger.warning(f"⚠️ Could not generate alternative command, skipping retry")
                                break  # 无法生成新命令，停止重试

                        # 根据最终结果分类存储
                        command_record = {
                            "round": probe_round,
                            "command": command,
                            "result": exec_result,
                            "focus_area": focus_area,
                            "analysis": analysis,
                            "retry_used": retry_used
                        }

                        if is_success:
                            # 记录成功的命令（只保存命令、round和iteration，不保存结果）
                            self.successful_commands.append({
                                "command": command, 
                                "round": probe_round,
                                "iteration": current_iteration
                            })
                            self.agent_logger.info(f"✅ Command successful{' (after retry)' if retry_used else ''}")

                            # 只有成功的结果才保存到Memory
                            if self.memory_manager:
                                raw_item = RawContextItem(
                                    source_agent=self.agent_type,
                                    source_agent_id=self.session_id or "",
                                    round_number=probe_round,
                                    raw_output=exec_result,
                                    command=command,
                                    execution_time=0.0,
                                    success=True,
                                    metadata={
                                        "session_id": self.session_id,
                                        "source_agent": self.agent_type.value,
                                        "iteration": current_iteration,
                                        "round_number": probe_round,
                                        "command": command,
                                        "focus_area": focus_area,
                                        "result": exec_result,
                                        "retry_used": retry_used
                                    }
                                )

                                self.memory_manager.add_item(raw_item, self.agent_type)

                            # 成功后重置重试计数
                            retry_count = 0
                            
                            # 保存前两个ITERATION的所有成功结果作为baseline context
                            if current_iteration <= 2:
                                # 格式化保存：命令 + 完整结果（不截断）
                                baseline_entry = f"\n### Iteration {current_iteration}, Round {probe_round}\n**Command**: {command}\n**Result**:\n{exec_result}\n"
                                self.baseline_context += baseline_entry
                                self.agent_logger.info(f"📝 Saved to baseline context (Iter {current_iteration}, Round {probe_round})")
                        else:
                            # 记录失败的命令（包含错误信息，用于t-1显示）
                            # 避免重复记录：检查最后一条记录是否已经是当前命令
                            if not self.failed_commands or self.failed_commands[-1]["command"] != command:
                                self.failed_commands.append({"command": command, "result": exec_result})
                            self.agent_logger.warning(f"⚠️ Command failed after all retries")
                            # 提取错误信息显示
                            error_match = re.search(r'Error:\s*(.+?)(?:\n|$)', exec_result)
                            if error_match:
                                self.agent_logger.error(f"Error: {error_match.group(1)}")

                        # 显示结果预览
                        result_preview = exec_result[:200] + "..." if len(exec_result) > 200 else exec_result
                        self.agent_logger.info(f"📊 Result preview: {result_preview}")

                    except Exception as e:
                        self.agent_logger.error(f"❌ Execution error: {e}")

                        # 尝试重试
                        if retry_count < max_retries_per_round:
                            retry_count += 1
                            self.agent_logger.warning(
                                f"⚠️ Exception occurred, attempting retry {retry_count}/{max_retries_per_round}")

                            # 记录失败（包含错误信息，用于t-1显示）
                            self.failed_commands.append({"command": command, "result": f"Error: {str(e)}"})

                            # 跳过当前轮次，让下一轮尝试不同的方法
                            probe_round += 1
                            retry_count = 0
                            continue

                        exec_result = f"Error: {str(e)}"
                        is_success = False

                        # 记录失败（包含错误信息，用于t-1显示）
                        self.failed_commands.append({"command": command, "result": exec_result})

                # 记录结果
                round_info = {
                    "round": probe_round,
                    "command": command,
                    "result": exec_result if exec_result else "",  # 不截断结果
                    "success": is_success,
                    "next_action": next_action,
                    "focus_area": focus_area,
                    "timestamp": datetime.now().isoformat(),
                    "retry_used": retry_used
                }
                all_results.append(round_info)

                # 检查是否完成
                if next_action == "COMPLETE":
                    self.agent_logger.success(f"✨ Probe completed at round {probe_round}")
                    break

                # 前进到下一轮
                probe_round += 1
                retry_count = 0  # 重置重试计数

                # 检查是否达到最大轮次
                if probe_round > self.max_iterations:
                    self.agent_logger.warning(f"⏱️ Reached maximum iterations ({self.max_iterations})")
                    break

            except Exception as e:
                self.agent_logger.error(f"💥 Error in probe round {probe_round}: {e}")
                error_info = {
                    "round": probe_round,
                    "error": str(e),
                    "next_action": "ERROR",
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(error_info)
                break

        # 只返回成功的命令历史作为probe_history（不截断）
        successful_history = "\n".join([
            f"Round {item['round']}: {item['command']}\nResult: {item.get('result', '')}"
            for item in all_results if item.get('success', False)
        ])

        # 保存baseline_context到memory（如果有且当前iteration <= 2）
        if self.baseline_context and current_iteration <= 2 and self.memory_manager and session_id:
            baseline_item = BaselineContextItem(
                session_id=session_id,
                iteration_numbers=[i for i in range(1, current_iteration + 1)],
                baseline_content=self.baseline_context,
                commands_included=[cmd["command"] for cmd in self.successful_commands if cmd.get("iteration", 0) <= 2],
                metadata={
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "source_agent": "probe"
                }
            )
            self.memory_manager.add_item(baseline_item, self.agent_type)
            self.agent_logger.info(f"💾 Baseline context saved to memory for iterations {baseline_item.iteration_numbers}")

        # 返回完整结果
        return {
            "total_rounds": len(all_results),
            "completed": True,
            "results": all_results,
            "probe_history": successful_history,
            "successful_commands_list": self.successful_commands,  # 添加完整的成功命令列表
            "failed_commands_list": self.failed_commands,  # 添加完整的失败命令列表
            "successful_commands": len(self.successful_commands),
            "failed_commands": len(self.failed_commands),
            "final_status": all_results[-1]["next_action"] if all_results else "NO_RESULTS",
            "session_id": self.session_id,
            "retries_used": sum(1 for r in all_results if r.get("retry_used", False)),
            "baseline_context": self.baseline_context  # 返回baseline_context
        }

    def reset(self):
        """重置探测器状态"""
        super().reset()
        self.probe_results = []
        self.probe_round = 0
        # 不再重置以下内容，让它们在整个任务期间累积：
        # - self.successful_commands
        # - self.failed_commands
        # - self.baseline_context (前两个 iteration 的基础信息)