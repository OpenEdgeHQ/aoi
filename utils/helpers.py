import re
from datetime import datetime
from typing import Any, Dict, List

def parse_text(text: str, max_length: int = 500) -> str:
    """解析并截断文本"""
    if not text:
        return ""
    
    text = text.strip()
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def cleanup_kubernetes_yaml(yaml_content: str) -> str:
    """清理Kubernetes YAML内容"""
    if not yaml_content:
        return ""
    
    # 移除一些动态生成的字段
    lines = yaml_content.split('\n')
    cleaned_lines = []
    
    skip_fields = [
        'resourceVersion',
        'uid',
        'selfLink',
        'creationTimestamp',
        'generation',
        'managedFields'
    ]
    
    skip_current = False
    indent_level = 0
    
    for line in lines:
        # 计算缩进级别
        current_indent = len(line) - len(line.lstrip())
        
        # 检查是否应该跳过这一行
        should_skip = False
        for field in skip_fields:
            if f'{field}:' in line:
                should_skip = True
                skip_current = True
                indent_level = current_indent
                break
        
        # 如果当前在跳过状态，检查缩进
        if skip_current:
            if current_indent > indent_level:
                continue  # 跳过子字段
            else:
                skip_current = False  # 结束跳过
        
        if not should_skip:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def format_timestamp(timestamp: str = None) -> str:
    """格式化时间戳"""
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
        except:
            dt = datetime.now()
    else:
        dt = datetime.now()
    
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def extract_error_from_state(state: str) -> Dict[str, Any]:
    """从状态中提取错误信息"""
    errors = {
        "has_error": False,
        "error_type": None,
        "error_message": None,
        "error_count": 0
    }
    
    # 检查常见错误模式
    error_patterns = [
        (r'Error:?\s*(.+)', 'generic'),
        (r'Failed:?\s*(.+)', 'failure'),
        (r'Exception:?\s*(.+)', 'exception'),
        (r'Crash.*', 'crash'),
        (r'Timeout.*', 'timeout'),
        (r'Connection refused', 'connection'),
        (r'Not found', 'not_found'),
        (r'Permission denied', 'permission'),
        (r'Invalid.*', 'invalid')
    ]
    
    state_lower = state.lower()
    
    for pattern, error_type in error_patterns:
        matches = re.findall(pattern, state_lower)
        if matches:
            errors["has_error"] = True
            errors["error_type"] = error_type
            errors["error_message"] = matches[0] if isinstance(matches[0], str) else str(matches[0])
            errors["error_count"] += len(matches)
    
    return errors

def merge_agent_decisions(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并多个智能体的决策"""
    merged = {
        "consensus": None,
        "decisions": decisions,
        "confidence": 0.0
    }
    
    if not decisions:
        return merged
    
    # 统计各种决策
    decision_counts = {}
    for decision in decisions:
        action = decision.get("action", "none")
        decision_counts[action] = decision_counts.get(action, 0) + 1
    
    # 找出最多的决策
    max_count = 0
    consensus_action = None
    for action, count in decision_counts.items():
        if count > max_count:
            max_count = count
            consensus_action = action
    
    merged["consensus"] = consensus_action
    merged["confidence"] = max_count / len(decisions) if decisions else 0
    
    return merged