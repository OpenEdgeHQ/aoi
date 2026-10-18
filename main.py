# main.py
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入日志
from utils.logger_config import AgentLogger, setup_logging, FileLogHandler

# 导入token限制工具
from utils.token_limiter import truncate_context, get_token_limiter

# 导入环境客户端
from environment.aiopslab_client import EnvironmentClient

# 导入智能体
from agents.observer_agent import ObserverAgent
from agents.probe_agent import ProbeAgent
from agents.executor_agent import ExecutorAgent
from agents.compressor_agent import CompressorAgent

# 导入Memory相关
from memory.memory_manager import MemoryManager
from memory.memory_item import (
    AgentType, SubTaskItem, RawContextItem,
    CompressedContextItem, TaskStatus, MemoryType
)

# 导入AWorld配置
from aworld.config.conf import AgentConfig


class AIOPlatform:
    """AI运维平台主类"""

    def __init__(self,
                 llm_config: AgentConfig,
                 env_client: Optional[EnvironmentClient] = None,
                 max_iterations: int = 6,
                 max_context_tokens: int = 25000,
                 max_output_tokens: int = 8000,
                 debug_no_submit: bool = False):
        """
        初始化AI运维平台

        Args:
            llm_config: LLM配置
            env_client: 环境客户端
            max_iterations: 最大迭代次数
            max_context_tokens: 最大上下文token数（防止超长）
            max_output_tokens: 最大输出token数
            debug_no_submit: Debug模式：不真正提交，只打印提交命令（session不会关闭）
        """
        # 初始化日志
        setup_logging()
        self.logger = AgentLogger("PLATFORM")

        # 配置
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.debug_no_submit = debug_no_submit

        # 环境客户端
        self.env_client = env_client
        self.session_id = None
        self.task_info = {}
        self.submit_format = {}

        # 初始化Memory管理器
        self.memory_manager = MemoryManager(
            max_raw_items=100000,
            max_compressed_items=10000,
            max_task_items=2000,
        )

        # 智能体将在run时初始化
        self.observer = None
        self.probe = None
        self.executor = None
        self.compressor = None

        # 执行状态
        self.current_iteration = 0
        self.execution_history = []
        
        # 存储problem_id
        self.problem_id = None

        # 结果收集
        self.execution_results = {
            "observer_outputs": [],
            "probe_results": [],
            "executor_results": [],
            "compressor_outputs": [],
            "iterations": []
        }
        
        # 评估结果（从AIOpsLab返回）
        self.evaluation_results = {}

    def _initialize_agents(self, task_info: Dict[str, Any]):
        """初始化所有智能体"""
        self.task_info = task_info

        # 获取submit格式
        self.submit_format = self.env_client.get_submit_format() if self.env_client else {}

        # 提取任务信息
        task_description = task_info.get('task_description', '')
        available_actions = task_info.get('available_actions', {})
        api_instruction = task_info.get('instructions', '')
        print("=" * 60)
        print(f"task_description: {task_description}")
        print("=" * 60)
        # 创建Observer - 它会自动初始化子任务队列，并从problem_id中提取任务类型
        self.observer = ObserverAgent(
            llm_config=self.llm_config,
            memory_manager=self.memory_manager,
            max_iterations=self.max_iterations,
            task_description=task_description,
            available_actions=available_actions,
            api_instruction=api_instruction,
            submit_format=self.submit_format,
            problem_id=self.problem_id  # 传递problem_id用于提取任务类型
        )

        # 创建Probe
        self.probe = ProbeAgent(
            llm_config=self.llm_config,
            memory_manager=self.memory_manager,
            max_iterations=3,
            task_description=task_description,
            available_actions=available_actions,
            api_instruction=api_instruction
        )

        # 创建Executor
        self.executor = ExecutorAgent(
            llm_config=self.llm_config,
            memory_manager=self.memory_manager,
            probe_agent=self.probe,
            max_iterations=1,
            use_probe=True,
            task_description=task_description,
            available_actions=available_actions,
            api_instruction=api_instruction
        )

        # 创建Compressor
        self.compressor = CompressorAgent(
            llm_config=self.llm_config,
            memory_manager=self.memory_manager,
            max_output_tokens=self.max_output_tokens,
            max_context_tokens=self.max_context_tokens
        )

        self.logger.info("✅ All agents initialized successfully")

        # 显示初始子任务队列
        self.logger.info(f"\n📋 Initial Task Queue:")
        for i, task in enumerate(self.observer.task_queue, 1):
            submit_marker = " [SUBMIT]" if task.is_submit_task else ""
            self.logger.info(
                f"  {i}. {task.task_name} "
                f"({task.target_agent.value if task.target_agent else 'unknown'})"
                f"{submit_marker}"
            )

    def execute_action(self, command: str) -> str:
        """执行命令"""
        if self.env_client:
            result = self.env_client.execute_action(command)
            if isinstance(result, dict):
                # 如果是提交动作，保存评估结果
                if result.get('is_submission') and result.get('evaluation'):
                    self.evaluation_results = result['evaluation']
                    self.logger.info(f"\n📊 Evaluation Results: {json.dumps(self.evaluation_results, indent=2)}")
                
                if result.get('error', False):
                    return f"Error: {result.get('result', str(result))}"
                return result.get('result', str(result))
            return str(result)
        else:
            return f"[Simulated execution] {command}"

    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """运行单次迭代 - 基于子任务队列"""
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"📍 ITERATION {iteration}/{self.max_iterations}")
        self.logger.info(f"{'=' * 80}")

        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "actions": []
        }

        try:
            # 获取当前子任务
            current_subtask = self.observer.get_current_subtask()

            if current_subtask:
                self.logger.info(f"\n📋 Current Subtask: {current_subtask.task_name}")
                self.logger.info(
                    f"   Target Agent: {current_subtask.target_agent.value if current_subtask.target_agent else 'Unknown'}")
                self.logger.info(f"   Objective: {current_subtask.task_objective}")

                # 如果是提交任务
                if current_subtask.is_submit_task:
                    self.logger.success(f"\n📮 Executing SUBMISSION task!")

                    # 提交任务也遵循相同原则：使用 previous_iteration_context
                    decision = await self.observer.analyze_and_decide(
                        compressed_context="",  # 不需要传入，使用 previous_iteration_context
                        iteration=iteration
                    )

                    # 使用Observer决定的提交命令
                    submission_command = decision.get("submission_command", "submit()")

                    # 执行提交
                    if self.env_client:
                        if self.debug_no_submit:
                            # Debug模式：只打印不提交
                            self.logger.warning(f"🔍 [DEBUG MODE] Would submit: {submission_command}")
                            self.logger.warning(f"🔍 [DEBUG MODE] Submission skipped - session remains open")
                            submit_result = {"status": "DEBUG_SKIP", "message": "Submission skipped in debug mode"}
                        else:
                            # 正常模式：真正提交
                            submit_result = self.execute_action(submission_command)
                            self.logger.info(f"[SUBMIT] Command: {submission_command}")
                            self.logger.info(f"[SUBMIT] Result: {submit_result}")

                        # 记录提交结果
                        iteration_data["actions"].append({
                            "type": "submit",
                            "command": submission_command,
                            "result": str(submit_result)
                        })

                        if (isinstance(submit_result, dict) and submit_result.get("error")) or str(submit_result).startswith("Error:"):
                            self.observer.previous_iteration_context = str(submit_result)
                            self.execution_results["iterations"].append(iteration_data)
                            return iteration_data

                        # 标记任务完成
                        if current_subtask:
                            current_subtask.complete_execution(True, f"Submitted: {submission_command}")
                            self.memory_manager.update_item(current_subtask, AgentType.OBSERVER)

                        # Debug模式：不结束，继续执行
                        if self.debug_no_submit:
                            self.logger.warning(f"🔍 [DEBUG MODE] Continuing execution (normally would have ended here)")
                            self.execution_results["iterations"].append(iteration_data)
                            # 不返回，让函数继续执行后面的正常流程
                        else:
                            # 正常模式：更准确地检查是否解决
                            if isinstance(submit_result, dict):
                                result_str = str(submit_result.get('result', submit_result))
                            else:
                                result_str = str(submit_result)

                            # 检查多种成功标志
                            if any(indicator in result_str for indicator in
                                   ["VALID_SUBMISSION", "VALID", "solved", "complete"]) or \
                                    self.env_client.is_problem_solved():
                                self.logger.success(f"\n✅ Problem SOLVED!")
                                self.execution_results["iterations"].append(iteration_data)
                                return {
                                    "status": "completed",
                                    "solution": submission_command,
                                    "iterations": iteration,
                                    "result": "VALID"
                                }
                            elif "INVALID_SUBMISSION" in result_str:
                                self.logger.warning(f"\n⚠️ Invalid submission - solution does not meet requirements")
                                self.execution_results["iterations"].append(iteration_data)
                                return {
                                    "status": "completed",
                                    "solution": submission_command,
                                    "iterations": iteration,
                                    "result": "INVALID"
                                }
                            else:
                                self.logger.info(f"\n📨 Submission received, status: {result_str[:100]}")
                                self.execution_results["iterations"].append(iteration_data)
                                return {
                                    "status": "completed",
                                    "solution": submission_command,
                                    "iterations": iteration,
                                    "result": "SUBMITTED"
                                }
            
            # 如果是debug模式且已经"提交"过，跳过后续处理
            if self.debug_no_submit and current_subtask and current_subtask.is_submit_task:
                return iteration_data

            # 1. Observer 分析并生成具体指令
            #    输入：上一轮的 compressed context（在 observer.previous_iteration_context 中）
            #    同时生成上一轮的总结
            self.logger.info(f"\n[OBSERVER] Analyzing and generating instructions")

            decision = await self.observer.analyze_and_decide(
                compressed_context="",  # 不需要传入，Observer 会从 previous_iteration_context 获取
                iteration=iteration
            )

            # 记录Observer决策
            self.execution_results["observer_outputs"].append({
                "iteration": iteration,
                "decision": decision,
                "timestamp": datetime.now().isoformat()
            })

            # 获取当前子任务（从决策中）
            current_subtask = decision.get('current_subtask')

            if not current_subtask:
                self.logger.error("No subtask available!")
                self.execution_results["iterations"].append(iteration_data)
                return {"status": "error", "error": "No subtask available"}

            # 输出决策信息
            self.logger.info(f"[OBSERVER] Decision: Activate {decision.get('next_agent', 'Unknown').upper()}")
            self.logger.info(f"[OBSERVER] Subtask: {current_subtask.task_name}")

            # 添加到执行历史
            self.observer.add_execution_result(
                agent_type="observer",
                action=f"Subtask: {current_subtask.task_name}",
                result=decision.get('reasoning', '')[:200],
                status="success"
            )

            # 3. 根据子任务目标执行相应智能体
            next_agent = decision.get("next_agent", "")
            task_instruction = decision.get("instruction", "")

            if next_agent == "complete" or decision.get("ready_to_submit", False):
                # 提交任务
                self.logger.success(f"\n✅ Executing submission!")

                # 使用Observer生成的提交命令（而不是硬编码）
                submission_command = decision.get('submission_command', 'submit()')

                if self.env_client:
                    if self.debug_no_submit:
                        # Debug模式：只打印不提交
                        self.logger.warning(f"🔍 [DEBUG MODE] Would submit: {submission_command}")
                        self.logger.warning(f"🔍 [DEBUG MODE] Submission skipped - session remains open")
                        submit_result = {"status": "DEBUG_SKIP", "message": "Submission skipped in debug mode"}
                    else:
                        # 正常模式：真正提交
                        submit_result = self.execute_action(submission_command)
                        self.logger.info(f"[SUBMIT] Command: {submission_command}")
                        self.logger.info(f"[SUBMIT] Result: {submit_result}")

                    iteration_data["actions"].append({
                        "type": "submit",
                        "command": submission_command,
                        "result": str(submit_result)
                    })

                    if (isinstance(submit_result, dict) and submit_result.get("error")) or str(submit_result).startswith("Error:"):
                        self.observer.previous_iteration_context = str(submit_result)
                        self.execution_results["iterations"].append(iteration_data)
                        return iteration_data

                    # 标记任务完成
                    if current_subtask:
                        current_subtask.complete_execution(True, f"Submitted: {submission_command}")
                        self.memory_manager.update_item(current_subtask, AgentType.OBSERVER)

                    # Debug模式：不结束，返回iteration_data以继续下一轮
                    if self.debug_no_submit:
                        self.logger.warning(f"🔍 [DEBUG MODE] Continuing execution (normally would have ended here)")
                        self.execution_results["iterations"].append(iteration_data)
                        return iteration_data  # 返回而不是continue，让外层循环继续
                    
                    # 正常模式：检查结果并返回
                    if "VALID" in str(submit_result) or self.env_client.is_problem_solved():
                        self.execution_results["iterations"].append(iteration_data)
                        return {
                            "status": "completed",
                            "solution": submission_command,
                            "iterations": iteration
                        }

                # 正常模式：返回完成状态
                self.execution_results["iterations"].append(iteration_data)
                return {
                    "status": "completed",
                    "solution": f"Task completed with: {submission_command}",
                    "iterations": iteration
                }


            elif next_agent == "probe":
                # 执行探测任务
                self.logger.info(f"\n[PROBE] Executing subtask: {current_subtask.task_name}")

                result = await self.probe.probe_run(
                    task_instruction=task_instruction,
                    execute_action=getattr(self.env_client, "execute_probe_action", self.execute_action),
                    current_subtask=current_subtask,
                    session_id=self.session_id
                )

                # 记录探测结果
                self.execution_results["probe_results"].append({
                    "iteration": iteration,
                    "subtask": current_subtask.task_name,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })

                iteration_data["actions"].append({
                    "type": "probe",
                    "subtask": current_subtask.task_name,
                    "rounds": result.get('total_rounds', 0),
                    "successful_commands": result.get('successful_commands', 0)
                })
                # 更新子任务状态
                if result.get('completed'):
                    current_subtask.complete_execution(True, f"Completed {result['total_rounds']} rounds")
                else:
                    current_subtask.execution_rounds += result.get('total_rounds', 1)

                self.memory_manager.update_item(current_subtask, AgentType.OBSERVER)

                # 输出结果摘要
                self.logger.info(f"[PROBE] Completed {result['total_rounds']} rounds")
                self.logger.info(
                    f"[PROBE] Success: {result['successful_commands']}, Failed: {result['failed_commands']}")

                # 添加到执行历史
                self.observer.add_execution_result(
                    agent_type="probe",
                    action=f"Probe: {current_subtask.task_name}",
                    result=result.get('probe_history', '')[:500],
                    status="success" if result.get('completed') else "partial"
                )

            elif next_agent == "executor":
                # 执行修复任务
                self.logger.info(f"\n[EXECUTOR] Executing subtask: {current_subtask.task_name}")
                
                # 获取executor_context（如果有）
                executor_context = decision.get("executor_context", "")
                if executor_context:
                    self.logger.info(f"[EXECUTOR] Received context from Observer ({len(executor_context)} chars)")

                result = await self.executor.executor_run(
                    task_instruction=task_instruction,
                    execute_action=self.execute_action,
                    current_subtask=current_subtask,
                    session_id=self.session_id,
                    executor_context=executor_context,
                    probe_execute_action=getattr(self.env_client, "execute_probe_action", None)
                )

                # 记录执行结果
                self.execution_results["executor_results"].append({
                    "iteration": iteration,
                    "subtask": current_subtask.task_name,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })

                iteration_data["actions"].append({
                    "type": "executor",
                    "subtask": current_subtask.task_name,
                    "rounds": result.get('total_rounds', 0),
                    "successful_commands": result.get('successful_commands', 0)
                })

                # 更新子任务状态
                if result.get('completed'):
                    current_subtask.complete_execution(True, f"Completed {result['total_rounds']} rounds")
                else:
                    current_subtask.execution_rounds += result.get('total_rounds', 1)

                self.memory_manager.update_item(current_subtask, AgentType.OBSERVER)

                # 输出结果摘要
                self.logger.info(f"[EXECUTOR] Completed {result['total_rounds']} rounds")
                self.logger.info(
                    f"[EXECUTOR] Success: {result['successful_commands']}, Failed: {result['failed_commands']}")

                # 添加到执行历史
                self.observer.add_execution_result(
                    agent_type="executor",
                    action=f"Executor: {current_subtask.task_name}",
                    result=result.get('execution_history', '')[:500],
                    status="success" if result.get('completed') else "partial"
                )

            # Iter n 执行完成后：
            # 压缩本轮 (iter n) 的 RAW_CONTEXT，保存为 compressed context 供下一轮使用
            if iteration < self.max_iterations:  # 不需要为最后一轮压缩
                try:
                    self.logger.info(f"\n[COMPRESSOR] Compressing iteration {iteration} RAW_CONTEXT")
                    compressed_context = await self.compressor.compressor_run(
                        session_id=self.session_id,
                        current_subtask=current_subtask
                    )
                    
                    # Limit token count
                    token_limiter = get_token_limiter(self.llm_config.llm_config.llm_model_name)
                    original_tokens = token_limiter.count_tokens(compressed_context)
                    
                    if original_tokens > self.max_context_tokens:
                        self.logger.warning(
                            f"⚠️  Context too long ({original_tokens} tokens), "
                            f"truncating to {self.max_context_tokens} tokens"
                        )

                        compressed_context = truncate_context(
                            compressed_context,
                            self.max_context_tokens,
                            self.llm_config.llm_config.llm_model_name
                        )
                        final_tokens = token_limiter.count_tokens(compressed_context)
                        self.logger.info(f"[COMPRESSOR] After truncation: {final_tokens} tokens")
                    else:
                        self.logger.info(f"[COMPRESSOR] Token count: {original_tokens} tokens (OK)")
                    
                    # 保存当前 iter n 的压缩上下文供 iter n+1 使用
                    self.observer.previous_iteration_context = compressed_context
                    self.logger.info(f"[COMPRESSOR] Saved iteration {iteration} compressed context for next iteration")
                    
                    # 记录压缩结果
                    self.execution_results["compressor_outputs"].append({
                        "iteration": iteration,
                        "compressed_context": compressed_context[:1000],
                        "size": len(compressed_context)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to compress context: {e}")
            
            # 前进到下一个任务
            self.observer.advance_to_next_task()

            self.execution_results["iterations"].append(iteration_data)
            return {"status": "continue"}

        except Exception as e:
            self.logger.error(f"Error in iteration {iteration}: {str(e)}")
            self.logger.error(traceback.format_exc())

            # 标记当前任务失败
            current_subtask = self.observer.get_current_subtask()
            if current_subtask:
                current_subtask.mark_failed(str(e))
                self.memory_manager.update_item(current_subtask, AgentType.OBSERVER)
                self.observer.advance_to_next_task()

            self.execution_results["iterations"].append(iteration_data)
            return {"status": "error", "error": str(e)}

    def log_token_usage(self):
        """显示token使用统计"""
        if not self.observer:
            return
            
        observer_tokens = self.observer.get_token_usage()
        probe_tokens = self.probe.get_token_usage() if self.probe else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        executor_tokens = self.executor.get_token_usage() if self.executor else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        total_input = observer_tokens["input_tokens"] + probe_tokens["input_tokens"] + executor_tokens["input_tokens"]
        total_output = observer_tokens["output_tokens"] + probe_tokens["output_tokens"] + executor_tokens["output_tokens"]
        total_all = observer_tokens["total_tokens"] + probe_tokens["total_tokens"] + executor_tokens["total_tokens"]
        
        self.logger.info(f"\n📊 Token Usage Statistics:")
        self.logger.info(f"  Observer   : {observer_tokens['total_tokens']:,} tokens (in: {observer_tokens['input_tokens']:,}, out: {observer_tokens['output_tokens']:,})")
        self.logger.info(f"  Probe      : {probe_tokens['total_tokens']:,} tokens (in: {probe_tokens['input_tokens']:,}, out: {probe_tokens['output_tokens']:,})")
        self.logger.info(f"  Executor   : {executor_tokens['total_tokens']:,} tokens (in: {executor_tokens['input_tokens']:,}, out: {executor_tokens['output_tokens']:,})")
        self.logger.info(f"  {'─' * 60}")
        self.logger.info(f"  Total      : {total_all:,} tokens (in: {total_input:,}, out: {total_output:,})")

    def save_execution_results(self, problem_id: str):
        """保存执行结果到JSON文件"""
        # 获取model名称（直接使用，不做替换）
        model_name = self.llm_config.llm_config.llm_model_name if hasattr(self, 'llm_config') and self.llm_config else "unknown"
        
        # 支持按轮次分开保存（通过 ROUND 环境变量）
        round_num = os.environ.get("ROUND", "")
        if round_num:
            res_dir = f"./res/{model_name}-round{round_num}"
        else:
            res_dir = f"./res/{model_name}"
        if getattr(self.env_client, "benchmark_name", None) == "ITBench":
            res_dir = f"./res/itbench/raw/{model_name}"
        os.makedirs(res_dir, exist_ok=True)

        # 生成文件名（使用problem_id）
        if getattr(self.env_client, "benchmark_name", None) == "ITBench":
            filename = f"{res_dir}/{self.session_id}.json"
        else:
            filename = f"{res_dir}/{problem_id}.json"

        # 收集 agent 的 token 使用情况
        token_usage = {
            "observer": self.observer.get_token_usage() if self.observer else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "probe": self.probe.get_token_usage() if self.probe else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "executor": self.executor.get_token_usage() if self.executor else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
        
        # 计算总计
        total_input_tokens = sum(agent_usage["input_tokens"] for agent_usage in token_usage.values())
        total_output_tokens = sum(agent_usage["output_tokens"] for agent_usage in token_usage.values())
        total_tokens = sum(agent_usage["total_tokens"] for agent_usage in token_usage.values())
        
        token_usage["total"] = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens
        }

        # 准备完整结果
        full_results = {
            "problem_id": problem_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "task_info": self.task_info,
            "token_usage": token_usage,  # 添加token使用统计
            "execution_results": self.execution_results,
            "evaluation_results": self.evaluation_results,  # 添加评估结果
            "final_task_queue": [
                {
                    "task_name": task.task_name,
                    "status": task.status.value,
                    "target_agent": task.target_agent.value if task.target_agent else None,
                    "execution_rounds": task.execution_rounds,
                    "is_submit_task": task.is_submit_task
                }
                for task in (self.observer.task_queue if self.observer else [])
            ]
        }

        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

        return filename

    async def run(self,
                  problem_id: Optional[str] = None,
                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整的问题解决流程

        Args:
            problem_id: 问题ID
            session_id: 会话ID

        Returns:
            执行结果
        """
        try:
            # 设置日志文件
            model_name = self.llm_config.llm_config.llm_model_name
            if problem_id:
                FileLogHandler.set_log_file(problem_id, model_name)
            elif session_id:
                FileLogHandler.set_log_file(f"session_{session_id}", model_name)

            self.logger.info("\n" + "=" * 80)
            self.logger.info("🚀 AI OPERATIONS PLATFORM STARTING")
            self.logger.info("=" * 80)

            # 保存problem_id
            self.problem_id = problem_id

            # 初始化或连接会话
            if self.env_client:
                if session_id:
                    result = self.env_client.connect_session(session_id)
                    self.logger.info(f"📌 Connected to session: {session_id}")
                elif problem_id:
                    # 重试init_problem，最多4次
                    result = None
                    init_failed = False
                    for attempt in range(5):
                        try:
                            result = self.env_client.init_problem(problem_id)
                            if result:
                                self.logger.info(f"📌 Initialized problem: {problem_id} (attempt {attempt + 1})")
                                break
                            else:
                                self.logger.warning(f"⚠️ Attempt {attempt + 1}: Empty task description, retrying...")
                                if attempt < 3:  # 不是最后一次尝试
                                    continue
                        except Exception as e:
                            self.logger.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                            if attempt < 3:  # 不是最后一次尝试
                                continue
                            else:
                                # 最后一次尝试也失败，标记初始化失败
                                init_failed = True
                                self.logger.error(f"❌ Failed to initialize problem {problem_id} after 4 attempts")
                                break
                    
                    # 检查初始化是否失败
                    if init_failed or not result or result.get('task_description', 'N/A') == 'N/A':
                        self.logger.error(f"❌ Terminating problem {problem_id} due to initialization failure")
                        self.logger.error(f"📝 Log saved, but result file will NOT be saved")
                        # 直接返回失败状态，不保存 res 文件
                        return {
                            "success": False,
                            "error": f"Failed to initialize problem {problem_id} after 4 attempts",
                            "session_id": None,
                            "initialization_failed": True  # 标记为初始化失败
                        }
                else:
                    raise ValueError("Either problem_id or session_id required")

                self.session_id = self.env_client.get_session_id()
                task_info = result

                # 显示任务信息
                self.logger.info(f"\n📋 Task Description:")
                task_desc = task_info.get('task_description', 'N/A')
                if len(task_desc) > 500:
                    self.logger.info(f"{task_desc[:500]}...")
                else:
                    self.logger.info(f"{task_desc}")

            else:
                # 模拟模式
                self.session_id = session_id or "test-session"
                task_info = {
                    "task_description": "Test task in simulation mode",
                    "available_actions": {},
                    "instructions": "Test instructions"
                }

            # 初始化智能体（包括创建子任务队列）
            self.logger.info(f"\n🤖 Initializing agents and task queue...")
            self._initialize_agents(task_info)

            # 主循环 - 执行子任务队列
            for iteration in range(1, self.max_iterations + 1):
                self.current_iteration = iteration

                # 运行单次迭代
                result = await self.run_iteration(iteration)

                # 检查是否完成
                if result.get("status") == "completed":
                    self.logger.info(f"\n{'=' * 80}")
                    self.logger.info(f"📊 TASK COMPLETED")
                    self.logger.info(f"{'=' * 80}")

                    # 显示最终任务队列状态
                    self.logger.info(f"\n📋 Final Task Queue Status:")
                    for i, task in enumerate(self.observer.task_queue, 1):
                        # 使用与ObserverAgent相同的状态图标逻辑
                        status_icon = {
                            TaskStatus.PENDING: "⏸",
                            TaskStatus.EXECUTING: "▶️",
                            TaskStatus.COMPLETED: "✅",
                            TaskStatus.FAILED: "❌",
                            TaskStatus.SKIPPED: "⏭"
                        }.get(task.status, "❓")
                        self.logger.info(f"  {status_icon} {task.task_name}")

                    # 显示token统计并保存结果
                    self.log_token_usage()
                    if problem_id:
                        result_file = self.save_execution_results(problem_id)
                        self.logger.info(f"📁 Results saved to: {result_file}")

                    # 清理会话
                    if self.env_client:
                        try:
                            self.env_client.cleanup_session()
                            self.logger.info("🧹 Session cleaned up successfully")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to cleanup session: {e}")

                    # 判断真正的成功：evaluation_results必须非空且success == true 或 "Detection Accuracy" == "Correct"
                    is_success = False
                    if self.evaluation_results:  # 确保evaluation_results不为空
                        is_success = (
                            self.evaluation_results.get('success') == True or
                            self.evaluation_results.get('Detection Accuracy') == 'Correct'
                        )
                    
                    if not is_success:
                        self.logger.warning(f"⚠️ Task completed but evaluation shows failure or empty results")

                    return {
                        "success": is_success,
                        "iterations": iteration,
                        "solution": result.get("solution", ""),
                        "session_id": self.session_id,
                        "evaluation_results": self.evaluation_results  # 添加评估结果
                    }

            # 达到最大迭代（应该在最后一轮提交）
            self.logger.warning(f"\n⏰ Reached maximum iterations")

            # 尝试最终提交
            if self.env_client:
                submit_result = self.execute_action("submit()")
                if self.env_client.is_problem_solved():
                    # 显示token统计并保存结果
                    self.log_token_usage()
                    if problem_id:
                        result_file = self.save_execution_results(problem_id)
                        self.logger.info(f"📁 Results saved to: {result_file}")

                    # 清理会话
                    if self.env_client:
                        try:
                            self.env_client.cleanup_session()
                            self.logger.info("🧹 Session cleaned up successfully")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to cleanup session: {e}")

                    # 判断真正的成功：evaluation_results必须非空且success == true 或 "Detection Accuracy" == "Correct"
                    is_success = False
                    if self.evaluation_results:  # 确保evaluation_results不为空
                        is_success = (
                            self.evaluation_results.get('success') == True or
                            self.evaluation_results.get('Detection Accuracy') == 'Correct'
                        )
                    
                    if not is_success:
                        self.logger.warning(f"⚠️ Task completed but evaluation shows failure or empty results")

                    return {
                        "success": is_success,
                        "iterations": self.max_iterations,
                        "message": "Solved at final submission",
                        "session_id": self.session_id,
                        "evaluation_results": self.evaluation_results  # 添加评估结果
                    }

            # 显示token统计并保存结果（即使失败）
            self.log_token_usage()
            if problem_id:
                result_file = self.save_execution_results(problem_id)
                self.logger.info(f"📁 Results saved to: {result_file}")

            return {
                "success": False,
                "iterations": self.max_iterations,
                "message": "Maximum iterations reached",
                "session_id": self.session_id,
                "evaluation_results": self.evaluation_results  # 添加评估结果
            }


        except Exception as e:
            self.logger.error(f"\n❌ Fatal error: {str(e)}")
            self.logger.error(traceback.format_exc())

            # 显示token统计并保存错误结果
            self.log_token_usage()
            if problem_id:
                try:
                    result_file = self.save_execution_results(problem_id)
                    self.logger.info(f"📁 Results saved to: {result_file}")
                except:
                    pass

            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "session_id": self.session_id,
                "evaluation_results": self.evaluation_results  # 添加评估结果
            }
        finally:
            # Close log file
            FileLogHandler.close()
