# agents/observer_agent.py
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re
import asyncio
import nest_asyncio

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig
from aworld.runner import Runners

from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager
from memory.memory_item import (
    AgentType, SubTaskItem,
    TaskStatus
)


from prompts.loader import get_prompt_loader


class ObserverAgent(BaseAgent):
    """观察者智能体 - 系统的主控制器和决策中心"""

    def __init__(self,
                 llm_config: AgentConfig,
                 memory_manager: MemoryManager,
                 max_iterations: int = 6,
                 task_description: str = "",
                 available_actions: Dict[str, str] = None,
                 api_instruction: str = "",
                 submit_format: Dict[str, Any] = None,
                 problem_id: Optional[str] = None):
        """
        初始化观察者智能体

        Args:
            llm_config: LLM配置
            memory_manager: 内存管理器
            max_iterations: 最大迭代次数
            task_description: 任务描述
            available_actions: 可用的API动作
            api_instruction: API使用说明格式
            submit_format: 提交格式信息
            problem_id: 问题ID（用于提取任务类型）
        """
        # 应用nest_asyncio以解决事件循环冲突
        nest_asyncio.apply()

        self.task_description = task_description
        self.available_actions = available_actions or {}
        self.api_instruction = api_instruction or ""
        self.submit_format = submit_format or {}
        self.prompt_loader = get_prompt_loader()
        self.problem_id = problem_id

        # 从problem_id中提取任务类型，而不是让LLM判断
        self.task_type = self._extract_task_type_from_problem_id(problem_id)

        super().__init__(
            name="Observer Agent",
            agent_type=AgentType.OBSERVER,
            llm_config=llm_config,
            memory_manager=memory_manager,
            max_iterations=max_iterations
        )

        # agent_logger 已在基类中初始化
        self.task_queue = []  # 子任务队列
        self.task_queue_ids = []  # 子任务ID列表
        self.current_task_index = 0  # 当前任务索引
        
        # 执行历史：记录每个agent的执行结果摘要（高层次时间线）
        # 与compressed_context不同，这是简要的行为列表，用于LLM快速了解"做了什么"
        self.execution_history = []
        
        # 上下文管理
        self.context_history_summaries = []  # 存储历史iter的compressed context总结
        self.previous_iteration_context = ""  # 存储上一轮的完整压缩上下文

        # 输出任务类型信息
        if self.task_type != "unknown":
            self.agent_logger.info(f"🔒 Task Type extracted from problem_id: {self.task_type}")

        # 初始化时创建初始子任务队列
        self._initialize_task_queue()

    def _extract_task_type_from_problem_id(self, problem_id: Optional[str]) -> str:
        """
        从problem_id中提取任务类型
        
        任务类型通过文件名中的关键词识别：
        - detection: 判断是否存在问题
        - localization: 定位具体的问题组件
        - analysis: 分析根因
        - mitigation: 修复问题
        
        Args:
            problem_id: 问题ID（如 "assign_to_non_existent_node_social_net-mitigation-1"）
            
        Returns:
            任务类型字符串（detection/localization/analysis/mitigation/unknown）
        """
        if not problem_id:
            return "unknown"
        
        problem_id_lower = problem_id.lower()
        
        # 按照优先级检查任务类型关键词
        if "-detection-" in problem_id_lower or problem_id_lower.endswith("-detection"):
            return "detection"
        elif "-localization-" in problem_id_lower or problem_id_lower.endswith("-localization"):
            return "localization"
        elif "-analysis-" in problem_id_lower or problem_id_lower.endswith("-analysis"):
            return "analysis"
        elif "-mitigation-" in problem_id_lower or problem_id_lower.endswith("-mitigation"):
            return "mitigation"
        else:
            # 如果没有找到标准关键词，尝试其他可能的变体
            if "detect" in problem_id_lower:
                return "detection"
            elif "local" in problem_id_lower:
                return "localization"
            elif "analy" in problem_id_lower or "root" in problem_id_lower:
                return "analysis"
            elif "mitigat" in problem_id_lower or "fix" in problem_id_lower or "repair" in problem_id_lower:
                return "mitigation"
            
        return "unknown"

    def _initialize_task_queue(self):
        """初始化子任务队列 - 基于LLM分析创建"""
        self.agent_logger.info("🎯 Initializing task queue...")

        try:
            # 使用LLM分析任务并生成初始子任务队列（从yaml加载prompt）
            initial_prompt = self.prompt_loader.get_prompt(
                agent_type="observer",
                prompt_type="task_queue_init",
                max_iterations=self.max_iterations,
                task_description=self.task_description,
                available_actions_preview=self._format_available_actions()[:1000]
            )

            # 调用LLM生成初始任务队列
            llm_response = asyncio.run(Runners.run(
                input=initial_prompt,
                agent=self.llm_agent
            ))

            # 解析响应
            response_text = llm_response.answer if hasattr(llm_response, 'answer') else str(llm_response)

            # 提取JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                task_data = json.loads(json_match.group())
                subtasks = task_data.get("subtasks", [])

                # 验证数量
                if len(subtasks) != self.max_iterations:
                    # 如果LLM没有生成正确数量，使用默认策略
                    subtasks = self._create_default_task_queue()
                else:
                    # 确保最后一个是提交任务
                    subtasks[-1]["is_submit"] = True
                    subtasks[-1]["name"] = "Submit Solution"
                    subtasks[-1]["target_agent"] = "observer"
            else:
                # 使用默认队列
                subtasks = self._create_default_task_queue()

            # 创建SubTaskItem对象
            for i, task_data in enumerate(subtasks, 1):
                subtask = SubTaskItem(
                    task_name=task_data.get("name", f"Task {i}"),
                    task_description=self.task_description,
                    task_objective=task_data.get("objective", ""),
                    target_agent=self._parse_agent_type(task_data.get("target_agent", "probe")),
                    priority=task_data.get("priority", 5),
                    iteration_number=i,
                    is_submit_task=task_data.get("is_submit", False),
                    task_context={
                        "iteration": i,
                        "total_iterations": self.max_iterations,
                        "task_description": self.task_description
                    },
                    max_rounds=5
                )

                # 添加到内存和队列
                self.memory_manager.add_item(subtask, self.agent_type)
                self.task_queue.append(subtask)
                self.task_queue_ids.append(subtask.id)

            self.agent_logger.info(f"✅ Created {len(self.task_queue)} subtasks in queue")

        except Exception as e:
            self.logger.error(f"Error creating task queue via LLM: {e}")
            # 使用默认队列
            self._create_and_store_default_queue()

    def _create_default_task_queue(self) -> List[Dict[str, Any]]:
        """创建默认的任务队列"""
        tasks = []

        # 根据max_iterations动态分配任务
        probe_count = max(1, self.max_iterations // 2)
        executor_count = max(0, self.max_iterations - probe_count - 1)

        # 探测任务
        for i in range(probe_count):
            tasks.append({
                "name": f"Investigate System State {i + 1}",
                "objective": "Gather diagnostic information",
                "target_agent": "probe",
                "priority": 10 - i,
                "is_submit": False
            })

        # 执行任务
        for i in range(executor_count):
            tasks.append({
                "name": f"Apply Fix {i + 1}",
                "objective": "Fix identified issues",
                "target_agent": "executor",
                "priority": 8 - i,
                "is_submit": False
            })

        # 提交任务
        tasks.append({
            "name": "Submit Solution",
            "objective": "Submit final solution",
            "target_agent": "observer",
            "priority": 10,
            "is_submit": True
        })

        return tasks

    def _create_and_store_default_queue(self):
        """创建并存储默认队列"""
        default_tasks = self._create_default_task_queue()

        for i, task_data in enumerate(default_tasks, 1):
            subtask = SubTaskItem(
                task_name=task_data["name"],
                task_description=self.task_description,
                task_objective=task_data["objective"],
                target_agent=self._parse_agent_type(task_data["target_agent"]),
                priority=task_data["priority"],
                iteration_number=i,
                is_submit_task=task_data["is_submit"],
                task_context={
                    "iteration": i,
                    "total_iterations": self.max_iterations
                },
                max_rounds=5
            )

            self.memory_manager.add_item(subtask, self.agent_type)
            self.task_queue.append(subtask)
            self.task_queue_ids.append(subtask.id)

    async def _update_task_queue(self, compressed_context: str, iteration: int) -> bool:
        """
        基于上下文更新任务队列（如有必要）

        Returns:
            是否更新了队列
        """
        # 只在前几轮考虑更新，后期保持稳定
        if iteration >= self.max_iterations - 1:
            return False

        # 关键约束：如果当前任务正在执行或失败，不应随意切换
        current_task = self.get_current_subtask()
        if current_task and current_task.status in [TaskStatus.EXECUTING, TaskStatus.FAILED]:
            self.agent_logger.info(
                f"🔒 Current task '{current_task.task_name}' is {current_task.status.value}. "
                f"Task queue update blocked - must complete or retry current task first."
            )
            return False

        try:
            # 使用LLM判断是否需要更新（从yaml加载prompt）
            update_prompt = self.prompt_loader.get_prompt(
                agent_type="observer",
                prompt_type="task_queue_update",
                iteration=iteration,
                max_iterations=self.max_iterations,
                compressed_context_preview=compressed_context[:2000],
                remaining_tasks=self._format_remaining_tasks()
            )

            llm_response = await Runners.run(
                input=update_prompt,
                agent=self.llm_agent
            )

            response_text = llm_response.answer if hasattr(llm_response, 'answer') else str(llm_response)

            # 解析JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                update_data = json.loads(json_match.group())

                if update_data.get("update_needed", False):
                    reason = update_data.get("reason", "")
                    self.agent_logger.info(f"📝 Updating task queue: {reason}")

                    updated_tasks = update_data.get("updated_tasks", [])
                    if updated_tasks:
                        # 更新剩余未执行的任务（保留最后的提交任务）
                        for i in range(iteration, min(len(self.task_queue) - 1, iteration + len(updated_tasks))):
                            if i < len(self.task_queue) - 1:  # 不更新最后的提交任务
                                task = self.task_queue[i]
                                update = updated_tasks[i - iteration] if i - iteration < len(updated_tasks) else {}

                                if update:
                                    task.task_name = update.get("name", task.task_name)
                                    task.task_objective = update.get("objective", task.task_objective)
                                    task.target_agent = self._parse_agent_type(update.get("target_agent", "probe"))
                                    task.priority = update.get("priority", task.priority)
                                    task.update()

                                    # 更新内存中的任务
                                    self.memory_manager.update_item(task, self.agent_type)

                        return True

        except Exception as e:
            self.logger.error(f"Error updating task queue: {e}")

        return False

    def _format_remaining_tasks(self) -> str:
        """格式化剩余任务列表"""
        remaining = []
        for i in range(self.current_task_index, len(self.task_queue)):
            task = self.task_queue[i]
            remaining.append(
                f"- Iteration {i + 1}: {task.task_name} ({task.target_agent.value if task.target_agent else 'unknown'})")
        return "\n".join(remaining) if remaining else "No remaining tasks"

    def get_current_subtask(self) -> Optional[SubTaskItem]:
        """获取当前应该执行的子任务"""
        if self.current_task_index < len(self.task_queue):
            return self.task_queue[self.current_task_index]
        return None

    def advance_to_next_task(self):
        """前进到下一个任务"""
        if self.current_task_index < len(self.task_queue):
            # 如果当前任务还在执行中，标记为完成
            current_task = self.task_queue[self.current_task_index]
            if current_task.status == TaskStatus.EXECUTING:
                current_task.complete_execution(True, "Advanced to next task")
                self.memory_manager.update_item(current_task, self.agent_type)

            self.current_task_index += 1
            self.agent_logger.info(f"📍 Advanced to task {self.current_task_index + 1}/{len(self.task_queue)}")

    def _initialize_llm_agent(self) -> Agent:
        """初始化LLM智能体"""
        return Agent(
            name=self.name,
            conf=self.llm_config,
            system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.prompt_loader.get_prompt(
            agent_type="observer",
            prompt_type="system",
            task_description=self.task_description,
            available_actions=self._format_available_actions(),
            api_instruction=self.api_instruction,
            submit_format=self._format_submit_info(),
            # task_type_info=self._get_task_type_display()
        )

    def _prepare_input(self, task_instruction: str, context: Dict[str, Any], **kwargs) -> str:
        """准备LLM输入"""
        # 获取当前子任务
        current_subtask = self.get_current_subtask()

        # 计算剩余轮数
        remaining_iterations = self.max_iterations - kwargs.get("iteration", self.current_iteration)
        
        # 当前迭代号
        current_iteration = kwargs.get("iteration", self.current_iteration)

        # 动态生成任务类型说明和提醒
        # task_type_display = self._get_task_type_display()
        
        # 生成简短的执行器提醒
        if self.task_type in ["detection", "localization", "analysis"]:
            executor_reminder = "NO executor needed (investigation only)."
        elif self.task_type == "mitigation":
            executor_reminder = "USE executor for repairs."
        else:
            executor_reminder = ""
        
        if current_iteration == 1:
            analysis_guidance = """**First Iteration - Initial Assessment**:
1. **Understand the task type**: The task type has been automatically determined from the problem ID
2. **Assess what you know**: What do you currently understand about the system?
3. **Decide next action**: What information or action do you need first?"""
        else:
            analysis_guidance = """**Ongoing Iteration - Progress Check**:
1. **Review findings**: What have you learned? What gaps remain?
2. **Check progress**: Are you closer to achieving the task objective?
3. **Decide next action**: Continue investigating, start fixing, or submit?"""
        
        # 准备参数
        params = {
            "compressed_context": kwargs.get("compressed_context", ""),
            "subtask_queue_status": self._get_subtask_queue_status_detailed(),
            "formatted_current_subtask": self._format_current_subtask(current_subtask),
            "execution_history": self._format_execution_history(),
            "iteration_number": current_iteration,
            "max_iterations": self.max_iterations,
            "remaining_iterations": remaining_iterations,
            "iteration_warning": self._get_iteration_warning(remaining_iterations),
            "task_type": self.task_type.capitalize(),
            "executor_reminder": executor_reminder,
            "analysis_guidance": analysis_guidance
        }

        return self.prompt_loader.get_prompt(
            agent_type="observer",
            prompt_type="user",
            **params
        )

    def _get_subtask_queue_status_detailed(self) -> str:
        """获取详细的子任务队列状态"""
        lines = [
            f"Total Tasks: {len(self.task_queue)}",
            f"Current Task Index: {self.current_task_index + 1}",
            f"",
            "Task Queue:"
        ]

        for i, task in enumerate(self.task_queue):
            status_icon = {
                TaskStatus.PENDING: "⏸",
                TaskStatus.EXECUTING: "▶️",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭"
            }.get(task.status, "❓")

            current_marker = " 👈 CURRENT" if i == self.current_task_index else ""
            submit_marker = " [SUBMIT]" if task.is_submit_task else ""

            lines.append(
                f"{i + 1}. {status_icon} {task.task_name} "
                f"({task.target_agent.value if task.target_agent else 'unknown'})"
                f"{submit_marker}{current_marker}"
            )

            if task.status == TaskStatus.COMPLETED and task.result:
                lines.append(f"   Result: {task.result[:100]}...")
            elif task.status == TaskStatus.FAILED and task.error_message:
                lines.append(f"   Error: {task.error_message[:100]}...")

        return "\n".join(lines)

    def _process_decision(self,
                          decision: Dict[str, Any],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """处理决策结果（简化版）"""
        current_subtask = self.get_current_subtask()
        if not current_subtask:
            self.agent_logger.error("No current subtask available!")
            return {"next_agent": "complete", "status": "COMPLETE", "error": "No subtask available"}

        # 记录confidence
        confidence = decision.get("confidence", 0)
        self.agent_logger.info(f"📊 Confidence: {confidence}%")

        # 获取核心字段
        next_action = decision.get("next_action", {})
        submission_info = decision.get("submission", {})

        # 检查是否提交
        if current_subtask.is_submit_task or submission_info.get("ready_to_submit", False):
            self.agent_logger.success("✅ Ready to submit!")
            submission_command = submission_info.get("submission_command", "submit()")
            self.agent_logger.info(f"📮 Submission: {submission_command}")

            return {
                "next_agent": "complete",
                "instruction": "Submit solution",
                "ready_to_submit": True,
                "submission_command": submission_command,
                "status": "COMPLETE",
                "current_subtask": current_subtask,
                "previous_iteration_summary": decision.get("previous_iteration_summary", "")
            }

        # 获取next_agent
        next_agent = next_action.get("action", "probe")
        if next_agent == "submit":
            next_agent = "complete"
        if not next_agent:
            next_agent = current_subtask.target_agent.value if current_subtask.target_agent else "probe"

        # 构建指令
        instruction = f"""
## Subtask: {current_subtask.task_name}
{current_subtask.task_objective}

## Instruction
{next_action.get("instruction", "")}
"""

        # 标记子任务开始执行
        if current_subtask.status == TaskStatus.PENDING:
            current_subtask.start_execution(self.session_id or "observer")
            self.memory_manager.update_item(current_subtask, self.agent_type)

        self.agent_logger.info(f"🎯 Next: {next_agent} | Subtask: {current_subtask.task_name}")

        return {
            "next_agent": next_agent,
            "instruction": instruction,
            "ready_to_submit": False,
            "current_subtask": current_subtask,
            "status": "CONTINUE",
            "executor_context": next_action.get("executor_context", ""),
            "previous_iteration_summary": decision.get("previous_iteration_summary", ""),
            "next_action": next_action,
            "submission": submission_info
        }

    async def analyze_and_decide(self,
                                 compressed_context: str,
                                 iteration: int = 1) -> Dict[str, Any]:
        """
        分析并做出决策 - 基于子任务队列
        
        在 iter n 开始时调用：
        - 输入：历史总结（iter 1 到 n-2）+ iter n-1 的完整压缩上下文
        - 输出：决策 + iter n-1 的总结

        Args:
            compressed_context: 压缩后的上下文（已废弃，使用 previous_iteration_context）
            iteration: 当前迭代轮数

        Returns:
            决策结果字典
        """
        self.current_iteration = iteration

        # 确保iteration对应正确的任务索引
        self.current_task_index = min(iteration - 1, len(self.task_queue) - 1)
        
        # 构建完整的上下文：历史总结 + 上一轮完整内容
        full_context = self._build_context_with_history("", iteration)

        # 考虑更新任务队列（非必须）
        if iteration > 1 and iteration < self.max_iterations - 1 and self.previous_iteration_context:
            updated = await self._update_task_queue(self.previous_iteration_context, iteration)
            if updated:
                self.agent_logger.info("📝 Task queue updated based on context")

        # 获取当前子任务
        current_subtask = self.get_current_subtask()

        if not current_subtask:
            self.agent_logger.error("No subtask available for current iteration!")
            return {
                "next_agent": "complete",
                "status": "COMPLETE",
                "ready_to_submit": True,
                "submission_command": "submit()"
            }

        self.agent_logger.info(f"📋 Current subtask: {current_subtask.task_name}")

        # 如果是提交任务，直接返回提交决策
        if current_subtask.is_submit_task:
            return {
                "next_agent": "complete",
                "instruction": "Submit solution",
                "ready_to_submit": True,
                "submission_command": "submit()",
                "status": "COMPLETE",
                "current_subtask": current_subtask
            }

        # 调用LLM生成具体指令（使用包含历史总结的完整上下文）
        output = await self.process(
            task_instruction="",
            context={},
            compressed_context=full_context,  # 使用包含历史总结的上下文
            current_subtask=current_subtask,
            iteration=iteration
        )

        # 确保返回字典格式
        if isinstance(output, str):
            output = self._parse_string_output(output)

        # 添加当前子任务到输出
        output["current_subtask"] = current_subtask

        # 处理上一轮的总结（如果是 iter > 1）
        if iteration > 1:
            prev_summary = output.get("previous_iteration_summary", "")
            if prev_summary:
                self.context_history_summaries.append(prev_summary)
                self.agent_logger.info(f"📝 Summary for Iter {iteration-1}: {len(prev_summary)} chars")
            else:
                default_summary = f"Iter {iteration-1}: [No summary provided]"
                self.context_history_summaries.append(default_summary)
                self.agent_logger.warning(f"⚠️ Using default summary")

        # 如果决策是调用executor，从LLM的输出中获取executor_context
        if output.get("next_agent") == "executor" or (
                current_subtask and current_subtask.target_agent == AgentType.EXECUTOR):
            # 从next_action中提取executor_context
            next_action = output.get("next_action", {})
            if isinstance(next_action, dict):
                executor_context = next_action.get("executor_context", "")
                if executor_context:
                    output["executor_context"] = executor_context
                    self.agent_logger.info(f"📎 Executor context provided ({len(executor_context)} chars)")
                else:
                    self.agent_logger.warning("⚠️ No executor_context provided by LLM")
                    output["executor_context"] = ""

        return output

    def mark_current_task_complete(self, success: bool = True, result: str = ""):
        """标记当前任务完成"""
        current_task = self.get_current_subtask()
        if current_task:
            current_task.complete_execution(success, result)
            self.memory_manager.update_item(current_task, self.agent_type)
            self.agent_logger.info(f"✅ Task completed: {current_task.task_name}")

    def _build_context_with_history(self, current_context: str, iteration: int) -> str:
        """
        构建包含历史总结的完整上下文
        
        逻辑：在iter n时
        - 历史总结：iter 1 到 iter n-2
        - 完整内容：iter n-1 的完整压缩上下文
        - current_context：iter n 的实时上下文（如果有）
        
        Args:
            current_context: 当前iter的实时上下文（可能为空）
            iteration: 当前迭代号
            
        Returns:
            包含历史总结和上一轮完整上下文的组合字符串
        """
        if iteration == 1:
            # 第一轮，没有历史
            return current_context if current_context else ""
        
        # 构建上下文：历史总结(1到n-2) + 上一轮完整内容(n-1)
        context_parts = []
        
        # 添加历史总结（iter 1 到 iter n-2）
        if self.context_history_summaries:
            context_parts.append("## Historical Context Summaries")
            for i, summary in enumerate(self.context_history_summaries, 1):
                context_parts.append(f"\n### Iteration {i} Summary")
                # 确保 summary 是字符串（兼容不同模型的输出格式）
                if isinstance(summary, dict):
                    import json
                    context_parts.append(json.dumps(summary, ensure_ascii=False))
                else:
                    context_parts.append(str(summary))
            context_parts.append("\n" + "="*80 + "\n")
        
        # 添加上一轮（iter n-1）的完整压缩上下文
        if self.previous_iteration_context:
            context_parts.append(f"## Previous Iteration ({iteration-1}) - Detailed Context")
            # 确保是字符串（兼容不同模型的输出格式）
            if isinstance(self.previous_iteration_context, dict):
                import json
                context_parts.append(json.dumps(self.previous_iteration_context, ensure_ascii=False))
            else:
                context_parts.append(str(self.previous_iteration_context))
        
        return "\n".join(context_parts)
    
    def _format_available_actions(self) -> str:
        """格式化可用动作"""
        if not self.available_actions:
            return "No available actions specified"

        formatted = []
        for action_name, action_desc in self.available_actions.items():
            # 保留完整的API文档，包括Args和Returns部分
            formatted.append(f"**{action_name}**: {action_desc}")

        return "\n\n".join(formatted)

    def _format_submit_info(self) -> str:
        """格式化提交信息"""
        if not self.submit_format:
            return "No submission format specified"

        return f"""
Session ID: {self.submit_format.get('session_id', 'N/A')}
Problem ID: {self.submit_format.get('problem_id', 'N/A')}
Submit API: {self.submit_format.get('submit_api', {}).get('api_name', 'N/A')}
Already Submitted: {self.submit_format.get('is_already_submitted', False)}
"""

    def _format_current_subtask(self, subtask: Optional[SubTaskItem]) -> str:
        """格式化当前子任务"""
        if not subtask:
            return "No current subtask"

        return f"""
Iteration: {subtask.iteration_number}
Name: {subtask.task_name}
Objective: {subtask.task_objective}
Target Agent: {subtask.target_agent.value if subtask.target_agent else 'Unknown'}
Status: {subtask.status.value}
Is Submit Task: {subtask.is_submit_task}
Execution Rounds: {subtask.execution_rounds}/{subtask.max_rounds}
"""

    def _format_execution_history(self) -> str:
        """格式化执行历史"""
        if not self.execution_history:
            return "No execution history yet"

        formatted = []
        for entry in self.execution_history[-5:]:
            formatted.append(f"""
Round {entry.get('round', '?')}:
Agent: {entry.get('agent', 'unknown')}
Action: {entry.get('action', 'N/A')}
Result: {entry.get('result', 'N/A')}
Status: {entry.get('status', 'unknown')}
""")

        return "\n---\n".join(formatted)

    def _get_iteration_warning(self, remaining_iterations: int) -> str:
        """获取迭代警告信息"""
        if remaining_iterations <= 1:
            return "⚠️ CRITICAL: This is the LAST iteration! Current task should be submission!"
        elif remaining_iterations == 2:
            return "⚠️ WARNING: Only 2 iterations left!"
        else:
            return f"ℹ️ {remaining_iterations} iterations remaining."

    def _parse_agent_type(self, agent_str: str) -> AgentType:
        """解析智能体类型字符串"""
        agent_str = agent_str.lower()
        if "probe" in agent_str:
            return AgentType.PROBE
        elif "executor" in agent_str:
            return AgentType.EXECUTOR
        elif "observer" in agent_str:
            return AgentType.OBSERVER
        else:
            return AgentType.PROBE

    def _parse_string_output(self, output: str) -> Dict[str, Any]:
        """解析字符串输出为字典（备用）"""
        return {
            "next_agent": "probe",
            "instruction": output,
            "status": "CONTINUE"
        }

    def add_execution_result(self, agent_type: str, action: str, result: str, status: str = "success"):
        """添加执行结果到历史"""
        self.execution_history.append({
            "round": self.current_iteration,
            "agent": agent_type,
            "action": action,
            "result": result,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
