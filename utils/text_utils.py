"""文本处理工具函数"""
from difflib import SequenceMatcher


def deduplicate_text(text: str, similarity_threshold: float = 0.93) -> tuple[str, dict]:
    """
    文本去重 - 使用difflib库去除重复或高度相似的行
    
    Args:
        text: 输入文本
        similarity_threshold: 相似度阈值（0-1之间），超过此阈值的行被视为重复
        
    Returns:
        tuple: (去重后的文本, 统计信息字典)
    """
    if not text or len(text.strip()) == 0:
        return text, {"original_length": 0, "deduplicated_length": 0, "reduction_ratio": 0.0}
    
    lines = text.split('\n')
    unique_lines = []
    
    for current_line in lines:
        # 跳过空行（但保留它们的位置信息）
        if not current_line.strip():
            unique_lines.append(current_line)
            continue

        # 检查当前行是否与已保存的行相似
        is_duplicate = False
        for saved_line in unique_lines:
            if not saved_line.strip():
                continue
                
            # 使用SequenceMatcher计算相似度
            similarity = SequenceMatcher(None, current_line, saved_line).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        # 如果不是重复行，添加到结果中
        if not is_duplicate:
            unique_lines.append(current_line)
    
    deduplicated_text = '\n'.join(unique_lines)
    
    # 计算统计信息
    original_length = len(text)
    deduplicated_length = len(deduplicated_text)
    reduction_ratio = (original_length - deduplicated_length) / original_length if original_length > 0 else 0.0
    
    stats = {
        "original_length": original_length,
        "deduplicated_length": deduplicated_length,
        "reduction_ratio": reduction_ratio
    }
    
    return deduplicated_text, stats

