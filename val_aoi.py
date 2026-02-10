#!/usr/bin/env python3
"""
验证和处理 AOI 结果文件

该脚本用于:
1. 读取 ./res/gpt-4o-mini/ 目录下的所有 .json 文件
2. 检查每个文件的 evaluation_results
3. 根据不同的条件判断结果是否正确
4. 删除没有 evaluation_results 的文件
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def check_detection_accuracy(eval_results: Dict[str, Any]) -> Optional[bool]:
    """
    检查 Detection Accuracy 是否为 Correct
    
    Args:
        eval_results: evaluation_results 字典
        
    Returns:
        如果有 Detection Accuracy 字段，返回是否为 "Correct"；否则返回 None
    """
    if "Detection Accuracy" in eval_results:
        return eval_results["Detection Accuracy"] == "Correct"
    return None


def check_success(eval_results: Dict[str, Any]) -> bool:
    """
    检查 success 字段是否为 true
    
    Args:
        eval_results: evaluation_results 字典
        
    Returns:
        success 字段的值，如果不存在返回 False
    """
    return eval_results.get("success", False)


def evaluate_result(eval_results: Dict[str, Any]) -> tuple[bool, str]:
    """
    评估结果是否正确
    
    Args:
        eval_results: evaluation_results 字典
        
    Returns:
        (是否正确, 判断依据的说明)
    """
    # 先检查是否有 Detection Accuracy 字段
    detection_accuracy = check_detection_accuracy(eval_results)
    
    if detection_accuracy is not None:
        return detection_accuracy, "Detection Accuracy"
    
    # 如果没有 Detection Accuracy，则检查 success
    success = check_success(eval_results)
    return success, "success"


def extract_task_type(data: Dict[str, Any]) -> Optional[str]:
    """
    从数据中提取任务类型
    
    Args:
        data: JSON 数据字典
        
    Returns:
        任务类型: 'detection', 'localization', 'analysis', 'mitigation' 或 None
    """
    # 从 problem_id 中提取任务类型
    problem_id = data.get("problem_id", "")
    
    task_types = ["detection", "localization", "analysis", "mitigation"]
    for task_type in task_types:
        if task_type in problem_id.lower():
            return task_type
    
    # 如果 problem_id 中没有，尝试从 evaluation_results 中查找
    eval_results = data.get("evaluation_results", {})
    for key in eval_results.keys():
        for task_type in task_types:
            if task_type in key.lower():
                return task_type
    
    return None


def process_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    处理单个 JSON 文件
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        处理结果字典，如果文件应该被删除则返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否有 evaluation_results 字段
        if "evaluation_results" not in data:
            print(f"⚠️  文件缺少 evaluation_results: {file_path.name}")
            return None
        
        eval_results = data["evaluation_results"]
        is_correct, criteria = evaluate_result(eval_results)
        
        # 提取任务类型
        task_type = extract_task_type(data)
        
        # 提取 token 信息
        in_tokens = eval_results.get("in_tokens", 0)
        out_tokens = eval_results.get("out_tokens", 0)
        total_tokens = in_tokens + out_tokens
        
        result = {
            "file": file_path.name,
            "is_correct": is_correct,
            "criteria": criteria,
            "task_type": task_type,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "total_tokens": total_tokens,
            "evaluation_results": eval_results
        }
        
        # 打印结果
        status = "✅ 正确" if is_correct else "❌ 错误"
        task_info = f"[{task_type}]" if task_type else ""
        print(f"{status} {task_info} [{criteria}] - {file_path.name}")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误: {file_path.name} - {e}")
        return None
    except Exception as e:
        print(f"❌ 处理文件时出错: {file_path.name} - {e}")
        return None


def delete_files(files_to_delete: list[Path], reason: str, dry_run: bool = True):
    """
    删除文件
    
    Args:
        files_to_delete: 要删除的文件列表
        reason: 删除原因描述
        dry_run: 如果为 True，只显示要删除的文件，不实际删除
    """
    if not files_to_delete:
        print(f"\n没有需要删除的文件（{reason}）。")
        return
    
    print(f"\n{'=' * 60}")
    print(f"发现 {len(files_to_delete)} 个文件（{reason}）:")
    for file_path in files_to_delete:
        print(f"  - {file_path.name}")
    
    if dry_run:
        print("\n[DRY RUN] 如要实际删除这些文件，请运行: python val_aoi.py --delete")
    else:
        confirm = input("\n确认删除这些文件？(y/N): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"✅ 已删除: {file_path.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 删除失败: {file_path.name} - {e}")
            print(f"\n成功删除 {deleted_count}/{len(files_to_delete)} 个文件")
        else:
            print("已取消删除操作。")


def main(delete: bool = False, delete_invalid: bool = False, delete_failed: bool = False):
    """
    主函数
    
    Args:
        delete: 如果为 True，删除所有问题文件（无效 + 失败）
        delete_invalid: 如果为 True，只删除无效文件（缺少 evaluation_results）
        delete_failed: 如果为 True，只删除失败文件（evaluation 显示错误）
    """
    import sys
    
    # 如果没有通过参数指定，检查命令行参数
    if not any([delete, delete_invalid, delete_failed]):
        delete = "--delete" in sys.argv
        delete_invalid = "--delete-invalid" in sys.argv or delete
        delete_failed = "--delete-failed" in sys.argv or delete
    
    # delete=True 时，同时删除无效和失败的文件
    if delete:
        delete_invalid = True
        delete_failed = True
    
    # dry_run: 只有明确指定删除时才实际删除
    dry_run = not (delete or delete_invalid or delete_failed)
    
    # Set target directory
    target_dir = Path("./res/anthropic/claude-sonnet-4.5")
    
    if not target_dir.exists():
        print(f"❌ 目录不存在: {target_dir}")
        return
    
    # 获取所有 JSON 文件
    json_files = list(target_dir.glob("*.json"))
    
    if not json_files:
        print(f"⚠️  目录下没有找到 JSON 文件: {target_dir}")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件\n")
    print("=" * 60)
    
    # 处理所有文件
    valid_results = []
    invalid_files = []  # 缺少 evaluation_results 的文件
    failed_files = []   # evaluation_results 显示失败的文件
    
    for json_file in json_files:
        result = process_json_file(json_file)
        if result is None:
            invalid_files.append(json_file)
        else:
            valid_results.append(result)
            # 如果结果是错误的，添加到失败文件列表
            if not result["is_correct"]:
                failed_files.append((json_file, result))
    
    # 统计结果
    print(f"\n{'=' * 60}")
    print("统计结果:")
    print(f"  总文件数: {len(json_files)}")
    print(f"  有效文件数: {len(valid_results)}")
    print(f"  无效文件数（缺少 evaluation_results）: {len(invalid_files)}")
    print(f"  失败文件数（evaluation 显示错误）: {len(failed_files)}")
    
    if valid_results:
        # 按 Detection Accuracy 分类统计
        detection_accuracy_results = [r for r in valid_results if r["criteria"] == "Detection Accuracy"]
        success_results = [r for r in valid_results if r["criteria"] == "success"]
        
        print(f"\n  按 Detection Accuracy 判断: {len(detection_accuracy_results)} 个")
        if detection_accuracy_results:
            correct_count = sum(1 for r in detection_accuracy_results if r["is_correct"])
            print(f"    - 正确: {correct_count}")
            print(f"    - 错误: {len(detection_accuracy_results) - correct_count}")
        
        print(f"\n  按 success 判断: {len(success_results)} 个")
        if success_results:
            correct_count = sum(1 for r in success_results if r["is_correct"])
            print(f"    - 正确: {correct_count}")
            print(f"    - 错误: {len(success_results) - correct_count}")
        
        # 按任务类型统计
        print(f"\n{'=' * 60}")
        print("按任务类型统计:")
        task_types = ["detection", "localization", "analysis", "mitigation"]

        for task_type in task_types:
            task_results = [r for r in valid_results if r.get("task_type") == task_type]
            if task_results:
                # 格式化 token 数量
                def format_tokens(tokens):
                    if tokens >= 1_000_000:
                        return f"{tokens/1_000_000:.2f}M"
                    elif tokens >= 1_000:
                        return f"{tokens/1_000:.2f}K"
                    else:
                        return str(tokens)
                
                # 统计 token 消耗
                total_tokens = sum(r["total_tokens"] for r in task_results)
                avg_tokens = total_tokens / len(task_results)
                
                # 所有任务类型都按原来的逻辑输出
                correct_count = sum(1 for r in task_results if r["is_correct"])
                accuracy = (correct_count / len(task_results)) * 100
                
                print(f"\n  {task_type.upper()}:")
                print(f"    - 总数: {len(task_results)}")
                print(f"    - 正确: {correct_count}")
                print(f"    - 错误: {len(task_results) - correct_count}")
                print(f"    - 准确率: {correct_count}/{len(task_results)} ({accuracy:.2f}%)")
                print(f"    - Token 消耗: {format_tokens(total_tokens)} (平均 {format_tokens(int(avg_tokens))})")
        
        # Analysis 任务额外输出详细评分点统计
        analysis_results = [r for r in valid_results if r.get("task_type") == "analysis"]
        if analysis_results:
            system_correct = 0
            fault_correct = 0
            for r in analysis_results:
                eval_res = r.get("evaluation_results", {})
                if eval_res.get("system_level_correct", False):
                    system_correct += 1
                if eval_res.get("fault_type_correct", False):
                    fault_correct += 1
            
            total_correct = system_correct + fault_correct
            total_count = len(analysis_results) * 2  # 每个任务2个评分点
            accuracy = (total_correct / total_count) * 100
            
            # 格式化 token 数量
            def format_tokens(tokens):
                if tokens >= 1_000_000:
                    return f"{tokens/1_000_000:.2f}M"
                elif tokens >= 1_000:
                    return f"{tokens/1_000:.2f}K"
                else:
                    return str(tokens)
            
            total_tokens = sum(r["total_tokens"] for r in analysis_results)
            avg_tokens = total_tokens / len(analysis_results)
            
            print(f"\n  ANALYSIS (详细评分点):")
            print(f"    - 总数: {len(analysis_results)} 个任务 ({total_count} 个评分点)")
            print(f"    - 正确: {total_correct} (system_level: {system_correct}, fault_type: {fault_correct})")
            print(f"    - 错误: {total_count - total_correct}")
            print(f"    - 准确率: {total_correct}/{total_count} ({accuracy:.2f}%)")
            print(f"    - Token 消耗: {format_tokens(total_tokens)} (平均 {format_tokens(int(avg_tokens))})")
        
        # 未识别任务类型的文件
        unknown_task_results = [r for r in valid_results if r.get("task_type") is None]
        if unknown_task_results:
            print(f"\n  未识别任务类型:")
            print(f"    - 总数: {len(unknown_task_results)}")
            correct_count = sum(1 for r in unknown_task_results if r["is_correct"])
            print(f"    - 正确: {correct_count}")
            print(f"    - 错误: {len(unknown_task_results) - correct_count}")
        
        # 总体准确率和token统计
        print(f"\n{'=' * 60}")
        total_correct = sum(1 for r in valid_results if r["is_correct"])
        accuracy = (total_correct / len(valid_results)) * 100 if valid_results else 0
        print(f"总体准确率: {total_correct}/{len(valid_results)} ({accuracy:.2f}%)")
        
        # 总 token 统计
        total_in_tokens = sum(r["in_tokens"] for r in valid_results)
        total_out_tokens = sum(r["out_tokens"] for r in valid_results)
        total_tokens = total_in_tokens + total_out_tokens
        
        def format_tokens(tokens):
            if tokens >= 1_000_000:
                return f"{tokens/1_000_000:.2f}M"
            elif tokens >= 1_000:
                return f"{tokens/1_000:.2f}K"
            else:
                return str(tokens)
        
        print(f"\n总 Token 消耗: {format_tokens(total_tokens)}")
        print(f"  - 输入: {format_tokens(total_in_tokens)}")
        print(f"  - 输出: {format_tokens(total_out_tokens)}")
        print(f"  - 平均每任务: {format_tokens(int(total_tokens/len(valid_results)))}")
    
    # 处理无效文件
    if delete_invalid:
        delete_files(invalid_files, "缺少 evaluation_results", dry_run=dry_run)
    else:
        if invalid_files:
            print(f"\n{'=' * 60}")
            print(f"发现 {len(invalid_files)} 个无效文件（缺少 evaluation_results）")
            print("使用 --delete-invalid 或 --delete 参数来删除这些文件")
    
    # 处理失败文件
    if delete_failed:
        failed_file_paths = [f[0] for f in failed_files]
        delete_files(failed_file_paths, "evaluation 显示错误", dry_run=dry_run)
    else:
        if failed_files:
            print(f"\n{'=' * 60}")
            print(f"发现 {len(failed_files)} 个失败文件（evaluation 显示错误）")
            print("失败文件列表:")
            for file_path, result in failed_files[:10]:  # 只显示前10个
                task_type = f"[{result['task_type']}]" if result['task_type'] else ""
                print(f"  - {task_type} {file_path.name}")
            if len(failed_files) > 10:
                print(f"  ... 还有 {len(failed_files) - 10} 个文件")
            print("\n使用 --delete-failed 或 --delete 参数来删除这些文件")
    
    print(f"\n{'=' * 60}")
    print("处理完成！")
    
    # 如果是通过函数参数调用，不显示命令行使用说明
    if not any([delete, delete_invalid, delete_failed]):
        print("\n使用说明:")
        print("  命令行方式:")
        print("    python val_aoi.py                    # 查看统计信息（不删除）")
        print("    python val_aoi.py --delete-invalid   # 删除无效文件（缺少 evaluation_results）")
        print("    python val_aoi.py --delete-failed    # 删除失败文件（evaluation 显示错误）")
        print("    python val_aoi.py --delete           # 删除所有问题文件（无效+失败）")
        print("")
        print("  代码调用方式:")
        print("    from val_aoi import main")
        print("    main(delete=True)                    # 删除所有问题文件")
        print("    main(delete_invalid=True)            # 只删除无效文件")
        print("    main(delete_failed=True)             # 只删除失败文件")


if __name__ == "__main__":
    main()

