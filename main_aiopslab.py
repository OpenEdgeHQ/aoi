# main_aiopslab.py
import asyncio
import json
import sys
import os

from typing import Dict, Any, Optional, List
import argparse
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


import llm_config

# ================================
# Development Configuration
# ================================
DEV_MODE = True  # Dev mode: True=use config below, False=use CLI args

# Evaluation configuration
DEV_EVALUATE_ALL = False  # True=evaluate all tasks (excluding DEV_EXCLUDE_PROBLEMS)
DEV_MAX_RETRIES = 1

# Specific tasks to run (effective when DEV_EVALUATE_ALL=False)
# All 86 tasks (excluding flower_* which have bugs)
DEV_SPECIFIC_PROBLEMS = [
    "k8s_target_port-misconfig-detection-1",
    "k8s_target_port-misconfig-localization-1",
    "k8s_target_port-misconfig-analysis-1",
    "k8s_target_port-misconfig-mitigation-1",
    "k8s_target_port-misconfig-detection-2",
    "k8s_target_port-misconfig-localization-2",
    "k8s_target_port-misconfig-analysis-2",
    "k8s_target_port-misconfig-mitigation-2",
    "k8s_target_port-misconfig-detection-3",
    "k8s_target_port-misconfig-localization-3",
    "k8s_target_port-misconfig-analysis-3",
    "k8s_target_port-misconfig-mitigation-3",
    "auth_miss_mongodb-detection-1",
    "auth_miss_mongodb-localization-1",
    "auth_miss_mongodb-analysis-1",
    "auth_miss_mongodb-mitigation-1",
    "revoke_auth_mongodb-detection-1",
    "revoke_auth_mongodb-localization-1",
    "revoke_auth_mongodb-analysis-1",
    "revoke_auth_mongodb-mitigation-1",
    "revoke_auth_mongodb-detection-2",
    "revoke_auth_mongodb-localization-2",
    "revoke_auth_mongodb-analysis-2",
    "revoke_auth_mongodb-mitigation-2",
    "user_unregistered_mongodb-detection-1",
    "user_unregistered_mongodb-localization-1",
    "user_unregistered_mongodb-analysis-1",
    "user_unregistered_mongodb-mitigation-1",
    "user_unregistered_mongodb-detection-2",
    "user_unregistered_mongodb-localization-2",
    "user_unregistered_mongodb-analysis-2",
    "user_unregistered_mongodb-mitigation-2",
    "misconfig_app_hotel_res-detection-1",
    "misconfig_app_hotel_res-localization-1",
    "misconfig_app_hotel_res-analysis-1",
    "misconfig_app_hotel_res-mitigation-1",
    "scale_pod_zero_social_net-detection-1",
    "scale_pod_zero_social_net-localization-1",
    "scale_pod_zero_social_net-analysis-1",
    "scale_pod_zero_social_net-mitigation-1",
    "assign_to_non_existent_node_social_net-detection-1",
    "assign_to_non_existent_node_social_net-localization-1",
    "assign_to_non_existent_node_social_net-analysis-1",
    "assign_to_non_existent_node_social_net-mitigation-1",
    "container_kill-detection",
    "container_kill-localization",
    "pod_failure_hotel_res-detection-1",
    "pod_failure_hotel_res-localization-1",
    "pod_kill_hotel_res-detection-1",
    "pod_kill_hotel_res-localization-1",
    "network_loss_hotel_res-detection-1",
    "network_loss_hotel_res-localization-1",
    "network_delay_hotel_res-detection-1",
    "network_delay_hotel_res-localization-1",
    "noop_detection_hotel_reservation-1",
    "noop_detection_social_network-1",
    "noop_detection_astronomy_shop-1",
    "astronomy_shop_ad_service_failure-detection-1",
    "astronomy_shop_ad_service_failure-localization-1",
    "astronomy_shop_ad_service_high_cpu-detection-1",
    "astronomy_shop_ad_service_high_cpu-localization-1",
    "astronomy_shop_ad_service_manual_gc-detection-1",
    "astronomy_shop_ad_service_manual_gc-localization-1",
    "astronomy_shop_cart_service_failure-detection-1",
    "astronomy_shop_cart_service_failure-localization-1",
    "astronomy_shop_image_slow_load-detection-1",
    "astronomy_shop_image_slow_load-localization-1",
    "astronomy_shop_kafka_queue_problems-detection-1",
    "astronomy_shop_kafka_queue_problems-localization-1",
    "astronomy_shop_loadgenerator_flood_homepage-detection-1",
    "astronomy_shop_loadgenerator_flood_homepage-localization-1",
    "astronomy_shop_payment_service_failure-detection-1",
    "astronomy_shop_payment_service_failure-localization-1",
    "astronomy_shop_payment_service_unreachable-detection-1",
    "astronomy_shop_payment_service_unreachable-localization-1",
    "astronomy_shop_product_catalog_service_failure-detection-1",
    "astronomy_shop_product_catalog_service_failure-localization-1",
    "astronomy_shop_recommendation_service_cache_failure-detection-1",
    "astronomy_shop_recommendation_service_cache_failure-localization-1",
    "redeploy_without_PV-detection-1",
    "redeploy_without_PV-analysis-1",
    "redeploy_without_PV-mitigation-1",
    "wrong_bin_usage-detection-1",
    "wrong_bin_usage-localization-1",
    "wrong_bin_usage-analysis-1",
    "wrong_bin_usage-mitigation-1",
]

# Problems to exclude (known bugs)
DEV_EXCLUDE_PROBLEMS = [
    "flower_node_stop",        # Bug: docker exec -it issue
    "flower_model_misconfig",  # Bug: docker exec -it issue
]

# Debug configuration
DEV_DEBUG_NO_SUBMIT = False  # True=don't submit, just print commands (session stays open)
# ================================

from utils.logger_config import AgentLogger, setup_logging, FileLogHandler
from environment.aiopslab_client import EnvironmentClient
from main import AIOPlatform

# Add AIOpsLab to Python path (for importing ProblemRegistry)
aiopslab_path = os.path.join(os.path.dirname(__file__), 'AIOpsLab')
if aiopslab_path not in sys.path:
    sys.path.insert(0, aiopslab_path)

from AIOpsLab.aiopslab.orchestrator.problems.registry import ProblemRegistry

# Create ProblemRegistry instance and get all problem IDs
_problem_registry = ProblemRegistry()
ALL_PROBLEM_IDS = _problem_registry.get_problem_ids()
DOCKER_DEPLOYMENT_PROBLEMS = _problem_registry.DOCKER_REGISTRY


class AIOpsLabEvaluator:
    """AIOpsLab Evaluator"""

    def __init__(self, 
                 llm_config: "llm_config.AgentConfig", 
                 server_host: str = "127.0.0.1", 
                 server_port: int = 8002,
                 max_context_tokens: int = 25000,
                 max_output_tokens: int = 8000,
                 debug_no_submit: bool = False):
        """
        Initialize the evaluator.

        Args:
            llm_config: LLM configuration
            server_host: Server host address
            server_port: Server port
            max_context_tokens: Maximum context token count
            max_output_tokens: Maximum output token count
            debug_no_submit: Debug mode: don't actually submit, just print commands
        """
        setup_logging()
        self.logger = AgentLogger("EVALUATOR")

        self.llm_config = llm_config
        self.server_host = server_host
        self.server_port = server_port
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.debug_no_submit = debug_no_submit

        # Set environment server
        EnvironmentClient.set_default_server(host=server_host, port=server_port)

        # Evaluation results
        self.results = []

    async def evaluate_problem(self, problem_id: str, max_retries: int = 1) -> Dict[str, Any]:
        """
        Evaluate a single problem.

        Args:
            problem_id: Problem ID
            max_retries: Maximum number of retries

        Returns:
            Evaluation result
        """
        # Set up per-problem log file
        model_name = self.llm_config.llm_config.llm_model_name
        FileLogHandler.set_log_file(problem_id, model_name)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"📝 EVALUATING PROBLEM: {problem_id}")
        self.logger.info(f"{'=' * 80}")

        start_time = datetime.now()
        best_result = None

        for attempt in range(max_retries):
            if attempt > 0:
                self.logger.info(f"\n🔄 Retry attempt {attempt}/{max_retries - 1}")
                await asyncio.sleep(2)  # 短暂延迟后重试

            try:
                # 创建新的客户端
                client = EnvironmentClient()

                # 创建平台实例
                platform = AIOPlatform(
                    llm_config=self.llm_config,
                    env_client=client,
                    max_iterations=14,
                    max_context_tokens=self.max_context_tokens,
                    max_output_tokens=self.max_output_tokens,
                    debug_no_submit=self.debug_no_submit
                )

                # 运行问题解决
                result = await platform.run(problem_id=problem_id)

                # 计算执行时间
                execution_time = (datetime.now() - start_time).total_seconds()

                # 记录结果
                eval_result = {
                    "problem_id": problem_id,
                    "success": result.get("success", False),
                    "iterations": result.get("iterations", 0),
                    "solution": result.get("solution", {}),
                    "error": result.get("error", None),
                    "execution_time": execution_time,
                    "session_id": result.get("session_id", None),
                    "attempt": attempt + 1,
                    "evaluation_results": result.get("evaluation_results", {})  # 添加评估结果
                }

                # 如果成功，直接返回
                if eval_result["success"]:
                    eval_results_str = ""
                    if eval_result.get("evaluation_results"):
                        eval_results_str = f"\n   - Evaluation: {json.dumps(eval_result['evaluation_results'])}"
                    
                    self.logger.success(
                        f"\n✅ Problem {problem_id} SOLVED\n"
                        f"   - Iterations: {eval_result['iterations']}\n"
                        f"   - Time: {execution_time:.2f}s\n"
                        f"   - Session: {eval_result['session_id']}{eval_results_str}\n"
                        f"   - Attempt: {attempt + 1}/{max_retries}"
                    )
                    FileLogHandler.close()
                    return eval_result

                # 保存最好的结果（用于失败时返回）
                if not best_result or eval_result.get("iterations", 0) > best_result.get("iterations", 0):
                    best_result = eval_result

                self.logger.warning(
                    f"\n⚠️ Problem {problem_id} not solved in attempt {attempt + 1}\n"
                    f"   - Iterations: {eval_result['iterations']}\n"
                    f"   - Time: {execution_time:.2f}s"
                )

            except Exception as e:
                self.logger.error(f"Exception evaluating {problem_id} (attempt {attempt + 1}): {str(e)}")
                if not best_result:
                    best_result = {
                        "problem_id": problem_id,
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "execution_time": (datetime.now() - start_time).total_seconds(),
                        "attempt": attempt + 1
                    }

        # 所有尝试都失败了
        if best_result:
            self.logger.error(
                f"\n❌ Problem {problem_id} FAILED after {max_retries} attempts\n"
                f"   - Best attempt: {best_result.get('iterations', 0)} iterations\n"
                f"   - Total time: {best_result['execution_time']:.2f}s"
            )
            FileLogHandler.close()
            return best_result

        FileLogHandler.close()
        return {
            "problem_id": problem_id,
            "success": False,
            "error": "All attempts failed",
            "execution_time": (datetime.now() - start_time).total_seconds()
        }

    async def evaluate_batch(self, problem_ids: List[str], max_retries: int = 1) -> Dict[str, Any]:
        """
        批量评估问题

        Args:
            problem_ids: 问题ID列表
            max_retries: 每个问题的最大重试次数

        Returns:
            批量评估结果
        """
        # 创建summary日志
        model_name = self.llm_config.llm_config.llm_model_name
        FileLogHandler.set_log_file("evaluation_summary", model_name)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"🚀 BATCH EVALUATION STARTING")
        self.logger.info(f"   Problems: {len(problem_ids)}")
        self.logger.info(f"   Max retries per problem: {max_retries}")
        self.logger.info(f"{'=' * 80}")

        results = []
        total_start = datetime.now()

        for i, problem_id in enumerate(problem_ids, 1):
            self.logger.info(f"\n[{i}/{len(problem_ids)}] Processing: {problem_id}")

            result = await self.evaluate_problem(problem_id, max_retries=max_retries)
            results.append(result)

            # 短暂延迟，避免过快请求
            if i < len(problem_ids):
                await asyncio.sleep(3)

        # 计算总执行时间
        total_time = (datetime.now() - total_start).total_seconds()

        # 统计结果
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total - successful

        # 计算平均值时处理除零情况
        avg_time = sum(r.get("execution_time", 0) for r in results) / total if total > 0 else 0
        avg_iterations = sum(
            r.get("iterations", 0) for r in results if r["success"]) / successful if successful > 0 else 0

        summary = {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "total_time": total_time,
            "average_time": avg_time,
            "average_iterations": avg_iterations,
            "results": results
        }

        # 重新切换到summary日志
        model_name = self.llm_config.llm_config.llm_model_name
        FileLogHandler.set_log_file("evaluation_summary", model_name)

        # 输出总结
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"📊 EVALUATION SUMMARY")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Total problems: {total}")
        self.logger.success(f"✅ Successful: {successful}")
        self.logger.error(f"❌ Failed: {failed}")
        self.logger.info(f"Success rate: {summary['success_rate']:.2%}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Average time per problem: {avg_time:.2f}s")
        if successful > 0:
            self.logger.info(f"Average iterations to solve: {avg_iterations:.2f}")

        # 输出成功的问题
        if successful > 0:
            self.logger.info(f"\n✅ Successful Problems:")
            for result in results:
                if result["success"]:
                    self.logger.success(
                        f"  - {result['problem_id']}: "
                        f"{result['iterations']} iterations, "
                        f"{result['execution_time']:.2f}s"
                    )

        # 输出失败的问题详情
        if failed > 0:
            self.logger.info(f"\n❌ Failed Problems:")
            for result in results:
                if not result["success"]:
                    error_msg = result.get('error', 'Unknown')
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    self.logger.error(f"  - {result['problem_id']}: {error_msg}")

        # 保存总结结果到JSON
        os.makedirs("./res", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        summary_file = f"./res/evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"\n📁 Summary saved to: {summary_file}")

        FileLogHandler.close()
        return summary

    async def evaluate_batch_multi_cluster(self, problem_ids: List[str], num_clusters: int = 5, 
                                          base_port: int = 8002, max_retries: int = 1) -> Dict[str, Any]:
        """
        使用多集群并发批量评估问题
        
        Args:
            problem_ids: 问题ID列表
            num_clusters: 集群数量
            base_port: 基础端口号（集群1使用base_port，集群2使用base_port+1，以此类推）
            max_retries: 每个问题的最大重试次数
            
        Returns:
            批量评估结果
        """
        # 创建summary日志
        model_name = self.llm_config.llm_config.llm_model_name
        FileLogHandler.set_log_file("evaluation_summary_multi_cluster", model_name)
        
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"🚀 MULTI-CLUSTER CONCURRENT EVALUATION STARTING")
        self.logger.info(f"   Problems: {len(problem_ids)}")
        self.logger.info(f"   Clusters: {num_clusters}")
        self.logger.info(f"   Max retries per problem: {max_retries}")
        self.logger.info(f"   Base port: {base_port}")
        self.logger.info(f"{'=' * 80}")
        
        # 将任务分配到不同的集群
        cluster_tasks = [[] for _ in range(num_clusters)]
        for i, problem_id in enumerate(problem_ids):
            cluster_idx = i % num_clusters
            cluster_tasks[cluster_idx].append(problem_id)
        
        # 显示任务分配
        self.logger.info(f"\n📋 Task Distribution:")
        for i, tasks in enumerate(cluster_tasks, 1):
            port = base_port + i - 1
            self.logger.info(f"  Cluster {i} (port {port}): {len(tasks)} tasks")
        
        total_start = datetime.now()
        
        # 定义在单个集群上执行任务的协程
        async def evaluate_on_cluster(cluster_id: int, tasks: List[str]):
            """在指定集群上评估任务"""
            port = base_port + cluster_id - 1
            cluster_results = []
            
            self.logger.info(f"\n🔹 Cluster {cluster_id} starting with {len(tasks)} tasks")
            
            for task_idx, problem_id in enumerate(tasks, 1):
                self.logger.info(f"  [Cluster {cluster_id}] [{task_idx}/{len(tasks)}] Processing: {problem_id}")
                
                try:
                    # 为每个问题设置单独的日志文件
                    FileLogHandler.set_log_file(problem_id, model_name)
                    
                    # 创建临时的评估器使用特定端口
                    temp_evaluator = AIOpsLabEvaluator(
                        llm_config=self.llm_config,
                        server_host=self.server_host,
                        server_port=port,
                        max_context_tokens=self.max_context_tokens,
                        max_output_tokens=self.max_output_tokens,
                        debug_no_submit=self.debug_no_submit
                    )
                    
                    result = await temp_evaluator.evaluate_problem(problem_id, max_retries=max_retries)
                    cluster_results.append(result)
                    
                    # 添加集群信息
                    result["cluster_id"] = cluster_id
                    result["cluster_port"] = port
                    
                except Exception as e:
                    self.logger.error(f"  [Cluster {cluster_id}] Error evaluating {problem_id}: {str(e)}")
                    cluster_results.append({
                        "problem_id": problem_id,
                        "success": False,
                        "error": str(e),
                        "cluster_id": cluster_id,
                        "cluster_port": port
                    })
            
            self.logger.info(f"  ✅ Cluster {cluster_id} completed {len(tasks)} tasks")
            return cluster_results
        
        # 并发执行所有集群的任务
        self.logger.info(f"\n⚡ Starting concurrent execution across {num_clusters} clusters...")
        
        cluster_coros = [
            evaluate_on_cluster(i + 1, cluster_tasks[i])
            for i in range(num_clusters)
            if cluster_tasks[i]  # 只执行有任务的集群
        ]
        
        all_cluster_results = await asyncio.gather(*cluster_coros)
        
        # 合并所有集群的结果
        results = []
        for cluster_results in all_cluster_results:
            results.extend(cluster_results)
        
        # 计算总执行时间
        total_time = (datetime.now() - total_start).total_seconds()
        
        # 统计结果
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful
        
        # 计算平均值
        avg_time = sum(r.get("execution_time", 0) for r in results) / total if total > 0 else 0
        avg_iterations = sum(
            r.get("iterations", 0) for r in results if r.get("success", False)
        ) / successful if successful > 0 else 0
        
        summary = {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "total_time": total_time,
            "average_time": avg_time,
            "average_iterations": avg_iterations,
            "num_clusters": num_clusters,
            "speedup_vs_serial": (avg_time * total) / total_time if total_time > 0 else 1.0,
            "results": results
        }
        
        # 重新切换到summary日志
        FileLogHandler.set_log_file("evaluation_summary_multi_cluster", model_name)
        
        # 输出总结
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"📊 MULTI-CLUSTER EVALUATION SUMMARY")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"Total problems: {total}")
        self.logger.info(f"Number of clusters: {num_clusters}")
        self.logger.success(f"✅ Successful: {successful}")
        self.logger.error(f"❌ Failed: {failed}")
        self.logger.info(f"Success rate: {summary['success_rate']:.2%}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Average time per problem: {avg_time:.2f}s")
        self.logger.info(f"⚡ Speedup vs serial: {summary['speedup_vs_serial']:.2f}x")
        if successful > 0:
            self.logger.info(f"Average iterations to solve: {avg_iterations:.2f}")
        
        # 按集群统计
        self.logger.info(f"\n📊 Per-Cluster Statistics:")
        for cluster_id in range(1, num_clusters + 1):
            cluster_res = [r for r in results if r.get("cluster_id") == cluster_id]
            if cluster_res:
                cluster_success = sum(1 for r in cluster_res if r.get("success", False))
                self.logger.info(
                    f"  Cluster {cluster_id}: {len(cluster_res)} tasks, "
                    f"{cluster_success} successful ({cluster_success/len(cluster_res)*100:.1f}%)"
                )
        
        # 输出成功的问题
        if successful > 0:
            self.logger.info(f"\n✅ Successful Problems:")
            for result in results:
                if result.get("success", False):
                    self.logger.success(
                        f"  - {result['problem_id']}: "
                        f"{result.get('iterations', 0)} iterations, "
                        f"{result.get('execution_time', 0):.2f}s "
                        f"(Cluster {result.get('cluster_id', '?')})"
                    )
        
        # 输出失败的问题详情
        if failed > 0:
            self.logger.info(f"\n❌ Failed Problems:")
            for result in results:
                if not result.get("success", False):
                    error_msg = result.get('error', 'Unknown')
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    self.logger.error(
                        f"  - {result['problem_id']}: {error_msg} "
                        f"(Cluster {result.get('cluster_id', '?')})"
                    )
        
        # 保存总结结果到JSON
        os.makedirs("./res", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        summary_file = f"./res/evaluation_summary_multi_cluster_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"\n📁 Summary saved to: {summary_file}")
        
        FileLogHandler.close()
        return summary

    def _get_existing_problem_ids(self) -> List[str]:
        """
        获取已存在的problem_ids（从res/{model}/目录中读取）
        
        Returns:
            已存在的problem_id列表
        """
        existing_ids = []
        model_name = self.llm_config.llm_config.llm_model_name
        # 直接使用模型名称
        res_dir = f"./res/{model_name}"
        
        if os.path.exists(res_dir):
            for filename in os.listdir(res_dir):
                if filename.endswith('.json'):
                    problem_id = filename[:-5]  # 去掉.json后缀
                    existing_ids.append(problem_id)
        
        return existing_ids

    async def evaluate_all_problems(self, max_retries: int = 1, exclude_problems: list = None) -> Dict[str, Any]:
        """
        评估所有注册的问题

        Args:
            max_retries: 最大重试次数
            exclude_problems: 要排除的问题前缀列表

        Returns:
            评估结果
        """
        # 获取所有问题ID
        all_problem_ids = ALL_PROBLEM_IDS.copy()
        
        # 应用排除列表（排除指定前缀的问题）
        if exclude_problems:
            excluded_count = 0
            filtered_ids = []
            for pid in all_problem_ids:
                if any(pid.startswith(prefix) for prefix in exclude_problems):
                    excluded_count += 1
                else:
                    filtered_ids.append(pid)
            all_problem_ids = filtered_ids
            self.logger.info(f"⛔ Excluded {excluded_count} problems matching: {exclude_problems}")
        
        # 获取已存在的problem_ids
        existing_problem_ids = self._get_existing_problem_ids()

        # 过滤掉已存在的problem_ids
        remaining_problem_ids = [i for i in all_problem_ids if i not in existing_problem_ids]
        
        # 显示当前使用的模型和结果目录
        model_name = self.llm_config.llm_config.llm_model_name
        
        self.logger.info(f"\n🤖 Model: {model_name}")
        self.logger.info(f"📁 Results directory: ./res/{model_name}/")
        self.logger.info(f"🔍 Found {len(all_problem_ids)} total problems in registry (after exclusions)")
        self.logger.info(f"📁 Skipping {len(existing_problem_ids)} already completed problems for this model")
        self.logger.info(f"🚀 Remaining problems to evaluate: {len(remaining_problem_ids)}")
        self.logger.info(f"\n📋 Problem list:")
        for i, pid in enumerate(remaining_problem_ids, 1):
            self.logger.info(f"  {i}. {pid}")

        # 批量评估所有问题
        return await self.evaluate_batch(remaining_problem_ids, max_retries=max_retries)

    async def evaluate_session(self, session_id: str) -> Dict[str, Any]:
        """
        评估已存在的会话

        Args:
            session_id: 会话ID

        Returns:
            评估结果
        """
        model_name = self.llm_config.llm_config.llm_model_name
        FileLogHandler.set_log_file(f"session_{session_id}", model_name)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"🔄 CONTINUING SESSION: {session_id}")
        self.logger.info(f"{'=' * 80}")

        start_time = datetime.now()

        try:
            # 创建新的客户端
            client = EnvironmentClient()

            # 创建平台实例
            platform = AIOPlatform(
                llm_config=self.llm_config,
                env_client=client,
                max_iterations=6,
                max_context_tokens=self.max_context_tokens,
                max_output_tokens=self.max_output_tokens,
                debug_no_submit=self.debug_no_submit
            )

            # 运行问题解决（连接现有会话）
            result = await platform.run(session_id=session_id)

            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds()

            # 记录结果
            eval_result = {
                "session_id": session_id,
                "success": result.get("success", False),
                "iterations": result.get("iterations", 0),
                "solution": result.get("solution", {}),
                "error": result.get("error", None),
                "execution_time": execution_time
            }

            if eval_result["success"]:
                self.logger.success(
                    f"\n✅ Session {session_id} COMPLETED\n"
                    f"   - Iterations: {eval_result['iterations']}\n"
                    f"   - Time: {execution_time:.2f}s"
                )
            else:
                self.logger.error(
                    f"\n❌ Session {session_id} FAILED\n"
                    f"   - Error: {eval_result.get('error', 'Unknown error')}\n"
                    f"   - Time: {execution_time:.2f}s"
                )

            FileLogHandler.close()
            return eval_result

        except Exception as e:
            self.logger.error(f"Exception in session {session_id}: {str(e)}")
            FileLogHandler.close()
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AIOpsLab Evaluation Tool")
    parser.add_argument("--api-source", type=str, choices=["openrouter", "openai"], 
                       help="API source: openrouter or openai")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--api-base", type=str, help="API base URL")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--problem", type=str, help="Single problem ID to evaluate")
    parser.add_argument("--problems", type=str, nargs="+", help="Multiple problem IDs to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all problems in registry")
    parser.add_argument("--session", type=str, help="Existing session ID to continue")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--retries", type=int, default=1, help="Max retries per problem")
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--max-context-tokens", type=int, default=25000, help="Max context tokens for Observer")
    parser.add_argument("--max-output-tokens", type=int, default=8000, help="Max output tokens")

    args = parser.parse_args()

    # 开发模式：使用代码中的配置
    if DEV_MODE:
        print("🔧 Running in DEV MODE - using configuration from llm_config.py")
        # 使用 llm_config 模块的默认配置
        api_source = llm_config.API_SOURCE
        api_key = llm_config.API_KEY
        api_base = llm_config.API_BASE
        model = llm_config.MODEL
        temperature = llm_config.TEMPERATURE
        max_context_tokens = llm_config.MAX_CONTEXT_TOKENS
        max_output_tokens = llm_config.MAX_OUTPUT_TOKENS
        evaluate_all = DEV_EVALUATE_ALL
        max_retries = DEV_MAX_RETRIES
        # 使用预先创建的 llm_config 对象
        llm_config_obj = llm_config.llm_config
    else:
        # 生产模式：使用命令行参数
        api_source = args.api_source
        if not api_source:
            raise ValueError("Please specify --api-source (openrouter or openai)")
        
        api_key = args.api_key or os.getenv(f"{api_source.upper()}_API_KEY")
        if not api_key:
            raise ValueError(f"{api_source.upper()}_API_KEY not found")
        
        api_base = args.api_base
        model = args.model
        temperature = args.temperature or 0.7
        max_context_tokens = args.max_context_tokens
        max_output_tokens = args.max_output_tokens
        evaluate_all = args.all
        max_retries = args.retries
        # 动态创建 llm_config 对象
        llm_config_obj = llm_config.create_llm_config(
            api_source=api_source,
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature
        )

    # 显示配置信息
    print(f"\n{'='*60}")
    print(f"🤖 Model: {model}")
    print(f"🔗 API Source: {api_source}")
    if api_base:
        print(f"🌐 API Base: {api_base}")
    print(f"🌡️  Temperature: {temperature}")
    print(f"🔢 Max Context Tokens: {max_context_tokens}")
    print(f"🔢 Max Output Tokens: {max_output_tokens}")
    if DEV_MODE and DEV_DEBUG_NO_SUBMIT:
        print(f"🔍 DEBUG MODE: Submission disabled (session will remain open)")
    print(f"📁 Results directory: ./res/{model}/")
    print(f"{'='*60}\n")

    # 创建评估器
    debug_no_submit = DEV_DEBUG_NO_SUBMIT if DEV_MODE else False
    evaluator = AIOpsLabEvaluator(
        llm_config=llm_config_obj,
        server_host=args.host,
        server_port=args.port,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        debug_no_submit=debug_no_submit
    )

    # Apply exclusion list
    if DEV_MODE and DEV_EVALUATE_ALL:
        evaluate_all = True
    # 执行评估

    if DEV_MODE and not DEV_EVALUATE_ALL and 'DEV_SPECIFIC_PROBLEMS' in globals():
        # 开发模式：运行指定的任务
        result = await evaluator.evaluate_batch(DEV_SPECIFIC_PROBLEMS, max_retries=max_retries)
    elif DEV_MODE and evaluate_all:
        # 开发模式：评估所有问题
        result = await evaluator.evaluate_all_problems(
            max_retries=max_retries,
            exclude_problems=DEV_EXCLUDE_PROBLEMS
        )
    elif args.all:
        # 评估所有问题
        result = await evaluator.evaluate_all_problems(max_retries=args.retries)
    elif args.session:
        # 评估现有会话
        result = await evaluator.evaluate_session(args.session)
    elif args.problems:
        # 批量评估
        result = await evaluator.evaluate_batch(args.problems, max_retries=args.retries)
    elif args.problem:
        # 单个问题评估
        result = await evaluator.evaluate_problem(args.problem, max_retries=args.retries)
    else:
        # 默认评估所有问题
        print("ℹ️ No specific problem specified, evaluating ALL problems in registry...")
        result = await evaluator.evaluate_all_problems(max_retries=args.retries)

    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n📁 Results saved to {args.output}")
    else:
        print(f"\n📊 Final Results:")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())