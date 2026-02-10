# utils/quick_problem_check.py
"""
快速问题检查工具 - 快速查看problem_id的故障注入方式和任务描述
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from problem_analyzer import ProblemAnalyzer

def quick_check(problem_id: str):
    """快速检查特定问题的故障注入方式"""
    print(f"🔍 快速检查问题: {problem_id}")
    print("=" * 60)

    try:
        analyzer = ProblemAnalyzer()
        analysis = analyzer.analyze_problem(problem_id)
        
        if 'error' in analysis:
            print(f"❌ 错误: {analysis['error']}")
            return
        
        # 显示关键信息
        print(f"📋 问题ID: {analysis['problem_id']}")
        print(f"🎯 任务类型: {analysis['task_type']}")
        print(f"⚠️  故障类型: {analysis['fault_type']}")
        print(f"🏗️  应用类型: {analysis['app_type']}")
        print(f"🚀 部署类型: {analysis['deployment_type']}")
        
        # 显示故障注入方式
        fault_method = analysis['fault_injection_method']
        print(f"\n🔧 故障注入方式:")
        print(f"   方法: {fault_method.get('method', 'Unknown')}")
        print(f"   描述: {fault_method.get('description', 'No description')}")
        if 'implementation' in fault_method:
            print(f"   具体实现: {fault_method['implementation']}")
        if 'target_service' in fault_method:
            print(f"   目标服务: {fault_method['target_service']}")
        if 'target_application' in fault_method:
            print(f"   目标应用: {fault_method['target_application']}")
        if 'namespace' in fault_method:
            print(f"   命名空间: {fault_method['namespace']}")
        
        # 显示任务描述
        task_desc = analysis['task_description']
        if task_desc and task_desc != "No task description available":
            print(f"\n📝 任务描述:")
            print(f"   {task_desc}")
        
        # 显示评估标准
        eval_criteria = analysis['evaluation_criteria']
        print(f"\n📊 评估标准:")
        print(f"   有评估方法: {eval_criteria.get('has_eval_method', False)}")
        print(f"   描述: {eval_criteria.get('description', 'No description')}")
        
        print("\n✅ 分析完成!")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def list_common_problems():
    """列出常见的问题类型"""
    print("📚 常见问题类型:")
    print("=" * 60)
    
    common_problems = [
        "k8s_target_port-misconfig-detection-1",
        "k8s_target_port-misconfig-localization-1", 
        "k8s_target_port-misconfig-analysis-1",
        "k8s_target_port-misconfig-mitigation-1",
        "auth_miss_mongodb-detection-1",
        "auth_miss_mongodb-localization-1",
        "auth_miss_mongodb-analysis-1",
        "auth_miss_mongodb-mitigation-1",
        "astronomy_shop_ad_service_failure-detection-1",
        "astronomy_shop_cart_service_failure-detection-1",
        "flower_node_stop-detection",
        "flower_model_misconfig-detection"
    ]
    
    for problem_id in common_problems:
        print(f"  • {problem_id}")

def main():
    """主函数 - 直接在IDE中运行"""

    # ===== 在这里修改你要检查的problem_id =====
    problem_id = "k8s_target_port-misconfig-detection-1"  # 修改这里！
    # ============================================
    
    # 常见问题列表
    common_problems = [
        "k8s_target_port-misconfig-detection-1",
        "k8s_target_port-misconfig-localization-1", 
        "k8s_target_port-misconfig-analysis-1",
        "k8s_target_port-misconfig-mitigation-1",
        "auth_miss_mongodb-detection-1",
        "auth_miss_mongodb-localization-1",
        "auth_miss_mongodb-analysis-1",
        "auth_miss_mongodb-mitigation-1",
        "astronomy_shop_ad_service_failure-detection-1",
        "astronomy_shop_cart_service_failure-detection-1",
        "flower_node_stop-detection",
        "flower_model_misconfig-detection"
    ]
    
    print("🔍 快速问题检查工具")
    print("=" * 60)
    print(f"当前检查的问题: {problem_id}")
    print("=" * 60)
    
    # 检查问题
    quick_check(problem_id)
    
    print("\n" + "=" * 60)
    print("📚 其他常见问题:")
    for pid in common_problems:
        if pid != problem_id:
            print(f"  • {pid}")
    
    print("\n💡 提示: 修改代码中的 problem_id 变量来检查其他问题")

if __name__ == "__main__":
    main()
