# utils/problem_analyzer.py
"""
问题分析工具 - 用于定位problem_id对应的注入错误方式和任务描述
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# 添加AIOpsLab路径到sys.path
aiopslab_path = Path(__file__).parent.parent / "AIOpsLab"
sys.path.insert(0, str(aiopslab_path))

try:
    from aiopslab.orchestrator.problems.registry import ProblemRegistry
    from aiopslab.orchestrator import Orchestrator
    from aiopslab.orchestrator.tasks.base import Task
except ImportError as e:
    print(f"导入AIOpsLab模块失败: {e}")
    print("请确保AIOpsLab目录存在且可访问")
    sys.exit(1)


class ProblemAnalyzer:
    """问题分析器 - 分析problem_id对应的故障注入方式和任务描述"""
    
    def __init__(self):
        self.registry = ProblemRegistry()
        self.orchestrator = None
        
    def get_available_problems(self) -> Dict[str, str]:
        """获取所有可用的问题ID和描述"""
        problems = {}
        for problem_id in self.registry.PROBLEM_REGISTRY.keys():
            try:
                # 获取问题实例
                problem_instance = self.registry.get_problem_instance(problem_id)
                
                # 获取任务类型
                task_type = self._extract_task_type(problem_id)
                
                # 获取故障类型
                fault_type = self._extract_fault_type(problem_id)
                
                # 获取应用类型
                app_type = self._extract_app_type(problem_id)
                
                problems[problem_id] = {
                    'task_type': task_type,
                    'fault_type': fault_type,
                    'app_type': app_type,
                    'description': f"{task_type} - {fault_type} - {app_type}"
                }
            except Exception as e:
                problems[problem_id] = {
                    'error': str(e),
                    'description': f"Error loading problem: {e}"
                }
        
        return problems
    
    def analyze_problem(self, problem_id: str) -> Dict[str, Any]:
        """分析特定problem_id的详细信息"""
        if problem_id not in self.registry.PROBLEM_REGISTRY:
            return {
                'error': f"Problem ID '{problem_id}' not found in registry",
                'available_problems': list(self.registry.PROBLEM_REGISTRY.keys())
            }
        
        try:
            # 获取问题实例
            problem_instance = self.registry.get_problem_instance(problem_id)
            
            # 分析问题
            analysis = {
                'problem_id': problem_id,
                'task_type': self._extract_task_type(problem_id),
                'fault_type': self._extract_fault_type(problem_id),
                'app_type': self._extract_app_type(problem_id),
                'deployment_type': self.registry.get_problem_deployment(problem_id),
                'fault_injection_method': self._analyze_fault_injection(problem_instance),
                'task_description': self._get_task_description(problem_instance),
                'available_actions': self._get_available_actions(problem_instance),
                'evaluation_criteria': self._get_evaluation_criteria(problem_instance)
            }
            
            return analysis
            
        except Exception as e:
            return {
                'problem_id': problem_id,
                'error': f"Error analyzing problem: {e}",
                'traceback': str(e)
            }
    
    def _extract_task_type(self, problem_id: str) -> str:
        """从problem_id提取任务类型"""
        if '-detection-' in problem_id:
            return 'Detection'
        elif '-localization-' in problem_id:
            return 'Localization'
        elif '-analysis-' in problem_id:
            return 'Analysis'
        elif '-mitigation-' in problem_id:
            return 'Mitigation'
        else:
            return 'Unknown'
    
    def _extract_fault_type(self, problem_id: str) -> str:
        """从problem_id提取故障类型"""
        # 常见的故障类型模式
        fault_patterns = {
            'k8s_target_port': 'Kubernetes Target Port Misconfiguration',
            'auth_miss_mongodb': 'MongoDB Authentication Missing',
            'revoke_auth': 'MongoDB Authentication Revoked',
            'user_unregistered': 'MongoDB User Unregistered',
            'misconfig_app': 'Application Misconfiguration',
            'scale_pod': 'Pod Scaling Issue',
            'assign_to_non_existent_node': 'Node Assignment Issue',
            'container_kill': 'Container Kill',
            'pod_failure': 'Pod Failure',
            'pod_kill': 'Pod Kill',
            'network_loss': 'Network Loss',
            'network_delay': 'Network Delay',
            'kernel_fault': 'Kernel Fault',
            'disk_woreout': 'Disk Woreout',
            'ad_service_failure': 'Advertisement Service Failure',
            'ad_service_high_cpu': 'Advertisement Service High CPU',
            'ad_service_manual_gc': 'Advertisement Service Manual GC',
            'cart_service_failure': 'Cart Service Failure',
            'image_slow_load': 'Image Slow Load',
            'kafka_queue_problems': 'Kafka Queue Problems',
            'loadgenerator_flood_homepage': 'Load Generator Flood Homepage',
            'payment_service_failure': 'Payment Service Failure',
            'payment_service_unreachable': 'Payment Service Unreachable',
            'product_catalog_failure': 'Product Catalog Failure',
            'recommendation_service_cache_failure': 'Recommendation Service Cache Failure',
            'redeploy_without_PV': 'Redeploy Without Persistent Volume',
            'wrong_bin_usage': 'Wrong Binary Usage',
            'operator_misoperation': 'Operator Misoperation',
            'flower_node_stop': 'Flower Node Stop',
            'flower_model_misconfig': 'Flower Model Misconfiguration'
        }
        
        for pattern, description in fault_patterns.items():
            if pattern in problem_id:
                return description
        
        return 'Unknown Fault Type'
    
    def _extract_app_type(self, problem_id: str) -> str:
        """从problem_id提取应用类型"""
        if 'social_net' in problem_id or 'social-network' in problem_id:
            return 'Social Network'
        elif 'hotel_res' in problem_id or 'hotel_reservation' in problem_id:
            return 'Hotel Reservation'
        elif 'astronomy_shop' in problem_id:
            return 'Astronomy Shop'
        elif 'flower' in problem_id:
            return 'Flower'
        else:
            return 'Unknown Application'
    
    def _analyze_fault_injection(self, problem_instance) -> Dict[str, Any]:
        """分析故障注入方式"""
        try:
            # 检查是否有inject_fault方法
            if hasattr(problem_instance, 'inject_fault'):
                # 尝试获取故障注入的详细信息
                fault_info = {
                    'method': 'inject_fault',
                    'description': 'Custom fault injection method'
                }
                
                # 如果是基础任务类，尝试获取更多信息
                if hasattr(problem_instance, 'app'):
                    fault_info['target_application'] = type(problem_instance.app).__name__
                
                if hasattr(problem_instance, 'faulty_service'):
                    fault_info['target_service'] = problem_instance.faulty_service
                
                if hasattr(problem_instance, 'namespace'):
                    fault_info['namespace'] = problem_instance.namespace
                
                return fault_info
            else:
                return {
                    'method': 'Unknown',
                    'description': 'No inject_fault method found'
                }
        except Exception as e:
            return {
                'method': 'Error',
                'description': f'Error analyzing fault injection: {e}'
            }
    
    def _get_task_description(self, problem_instance) -> str:
        """获取任务描述"""
        try:
            if hasattr(problem_instance, 'task_desc'):
                return problem_instance.task_desc
            elif hasattr(problem_instance, 'get_task_description'):
                return problem_instance.get_task_description()
            else:
                return "No task description available"
        except Exception as e:
            return f"Error getting task description: {e}"
    
    def _get_available_actions(self, problem_instance) -> Dict[str, str]:
        """获取可用动作"""
        try:
            if hasattr(problem_instance, 'actions'):
                return problem_instance.actions
            else:
                return {}
        except Exception as e:
            return {'error': str(e)}
    
    def _get_evaluation_criteria(self, problem_instance) -> Dict[str, Any]:
        """获取评估标准"""
        try:
            if hasattr(problem_instance, 'eval'):
                return {
                    'has_eval_method': True,
                    'description': 'Custom evaluation method available'
                }
            else:
                return {
                    'has_eval_method': False,
                    'description': 'No custom evaluation method'
                }
        except Exception as e:
            return {
                'has_eval_method': False,
                'error': str(e)
            }


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析AIOpsLab问题')
    parser.add_argument('--problem-id', type=str, help='要分析的问题ID')
    parser.add_argument('--list-all', action='store_true', help='列出所有可用问题')
    parser.add_argument('--search', type=str, help='搜索包含关键词的问题')
    
    args = parser.parse_args()
    
    analyzer = ProblemAnalyzer()
    args.list_all = True
    if args.list_all:
        print("=== 所有可用问题 ===")
        problems = analyzer.get_available_problems()
        for problem_id, info in problems.items():
            print(f"{problem_id}: {info.get('description', 'No description')}")
    
    elif args.search:
        print(f"=== 搜索包含 '{args.search}' 的问题 ===")
        problems = analyzer.get_available_problems()
        matching = {k: v for k, v in problems.items() if args.search.lower() in k.lower()}
        for problem_id, info in matching.items():
            print(f"{problem_id}: {info.get('description', 'No description')}")
    
    elif args.problem_id:
        print(f"=== 分析问题: {args.problem_id} ===")
        analysis = analyzer.analyze_problem(args.problem_id)
        
        if 'error' in analysis:
            print(f"错误: {analysis['error']}")
            if 'available_problems' in analysis:
                print("可用问题:")
                for pid in analysis['available_problems'][:10]:  # 只显示前10个
                    print(f"  - {pid}")
        else:
            print(f"问题ID: {analysis['problem_id']}")
            print(f"任务类型: {analysis['task_type']}")
            print(f"故障类型: {analysis['fault_type']}")
            print(f"应用类型: {analysis['app_type']}")
            print(f"部署类型: {analysis['deployment_type']}")
            print(f"故障注入方式: {analysis['fault_injection_method']}")
            print(f"任务描述: {analysis['task_description']}")
            print(f"评估标准: {analysis['evaluation_criteria']}")
    
    else:
        print("请指定 --problem-id, --list-all 或 --search 参数")
        print("使用 --help 查看详细帮助")


if __name__ == "__main__":
    main()
