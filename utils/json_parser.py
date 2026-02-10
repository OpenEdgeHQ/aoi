#!/usr/bin/env python3
"""
JSON日志解析工具
用于快速检查提交状态
"""

import json
from typing import Dict, Any, Optional


def check_submission_status(json_file_path: str) -> Dict[str, Any]:
    """
    检查JSON日志文件中的提交状态
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        包含提交状态信息的字典
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": f"文件不存在: {json_file_path}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON格式错误: {e}"}
    
    # 初始化状态信息
    status = {
        "is_submitted": False,
        "submission_result": None,
        "submission_command": None,
        "final_status": "unknown",
        "success": False
    }
    
    # 检查执行结果中的提交信息
    execution_results = data.get("execution_results", {})
    iterations = execution_results.get("iterations", [])
    
    # 查找最后一次提交
    for iteration in reversed(iterations):
        actions = iteration.get("actions", [])
        for action in reversed(actions):
            if action.get("type") == "submit":
                status["is_submitted"] = True
                status["submission_command"] = action.get("command")
                
                # 检查提交结果
                result = action.get("result", "")
                if isinstance(result, dict):
                    result_str = str(result.get("result", result))
                else:
                    result_str = str(result)
                
                # 判断提交状态
                if any(indicator in result_str for indicator in 
                      ["VALID_SUBMISSION", "VALID", "solved", "complete"]):
                    status["submission_result"] = "VALID"
                    status["final_status"] = "success"
                    status["success"] = True
                elif "INVALID_SUBMISSION" in result_str:
                    status["submission_result"] = "INVALID"
                    status["final_status"] = "failed"
                else:
                    status["submission_result"] = "SUBMITTED"
                    status["final_status"] = "pending"
                
                return status
    
    return status


def get_task_status_summary(json_file_path: str) -> str:
    """
    获取任务状态摘要
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        任务状态摘要字符串
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return "❌ 无法读取JSON文件"
    
    # 获取任务队列状态
    task_queue = data.get("final_task_queue", [])
    
    summary_lines = []
    for i, task in enumerate(task_queue, 1):
        status_icon = {
            "pending": "⏸",
            "executing": "▶️", 
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭"
        }.get(task.get("status", "unknown"), "❓")
        
        task_name = task.get("task_name", "Unknown")
        summary_lines.append(f"  {status_icon} {task_name}")
    
    return "\n".join(summary_lines)


def is_submission_successful(json_file_path: str) -> bool:
    """
    快速检查提交是否成功
    
    Args:
        json_file_path: JSON文件路径
        
    Returns:
        True如果提交成功，False否则
    """
    status = check_submission_status(json_file_path)
    return status.get("success", False)


# 使用示例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python json_parser.py <json_file_path>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # 检查提交状态
    status = check_submission_status(json_file)
    print(f"📊 提交状态检查:")
    print(f"  是否已提交: {'✅' if status['is_submitted'] else '❌'}")
    if status['is_submitted']:
        print(f"  提交命令: {status['submission_command']}")
        print(f"  提交结果: {status['submission_result']}")
        print(f"  最终状态: {status['final_status']}")
        print(f"  是否成功: {'✅' if status['success'] else '❌'}")
    
    # 显示任务状态摘要
    print(f"\n📋 任务状态摘要:")
    print(get_task_status_summary(json_file))
