#!/usr/bin/env python3
"""
提交状态检查示例
演示如何从JSON日志文件中检查提交是否通过
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.json_parser import check_submission_status, get_task_status_summary, is_submission_successful


def main():
    """主函数示例"""
    
    # 示例JSON文件路径（请替换为实际的JSON文件路径）
    json_file_path = "./res/your_problem_id_20250101_120000.json"
    
    print("🔍 提交状态检查示例")
    print("=" * 50)
    
    # 方法1: 获取详细提交状态
    print("\n📊 方法1: 详细状态检查")
    status = check_submission_status(json_file_path)
    
    if "error" in status:
        print(f"❌ 错误: {status['error']}")
        return
    
    print(f"是否已提交: {'✅' if status['is_submitted'] else '❌'}")
    if status['is_submitted']:
        print(f"提交命令: {status['submission_command']}")
        print(f"提交结果: {status['submission_result']}")
        print(f"最终状态: {status['final_status']}")
        print(f"是否成功: {'✅' if status['success'] else '❌'}")
    
    # 方法2: 快速检查是否成功
    print(f"\n⚡ 方法2: 快速成功检查")
    success = is_submission_successful(json_file_path)
    print(f"提交是否成功: {'✅' if success else '❌'}")
    
    # 方法3: 获取任务状态摘要
    print(f"\n📋 方法3: 任务状态摘要")
    summary = get_task_status_summary(json_file_path)
    print(summary)
    
    # 方法4: 在代码中使用
    print(f"\n💻 方法4: 代码中使用示例")
    print("""
# 在你的代码中使用:
from utils.json_parser import check_submission_status, is_submission_successful

# 检查详细状态
status = check_submission_status("path/to/your/result.json")
if status['success']:
    print("✅ 提交成功!")
else:
    print("❌ 提交失败或未提交")

# 或者快速检查
if is_submission_successful("path/to/your/result.json"):
    print("✅ 提交成功!")
else:
    print("❌ 提交失败或未提交")
    """)


if __name__ == "__main__":
    main()

