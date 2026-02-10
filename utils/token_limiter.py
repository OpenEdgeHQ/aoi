"""
Token 限制工具
用于防止上下文超过模型的最大token限制
"""

import tiktoken
from typing import Optional


class TokenLimiter:
    """Token限制器"""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        初始化Token限制器

        Args:
            model_name: 模型名称，用于选择合适的tokenizer
        """
        try:
            # 尝试获取模型对应的编码器
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # 如果模型不支持，使用默认的cl100k_base（GPT-4使用的编码）
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            token数量
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int, 
                     keep_start: bool = True,
                     keep_end: bool = True,
                     separator: str = "\n\n[... TRUNCATED ...]\n\n") -> str:
        """
        截断文本到指定的token数量
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            keep_start: 是否保留开头
            keep_end: 是否保留结尾
            separator: 截断分隔符
            
        Returns:
            截断后的文本
        """
        if not text:
            return ""
        
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # 计算分隔符的token数
        separator_tokens = len(self.encoding.encode(separator))
        available_tokens = max_tokens - separator_tokens
        
        if keep_start and keep_end:
            # 保留开头和结尾
            start_tokens = available_tokens // 2
            end_tokens = available_tokens - start_tokens
            
            start_text = self.encoding.decode(tokens[:start_tokens])
            end_text = self.encoding.decode(tokens[-end_tokens:])
            
            return f"{start_text}{separator}{end_text}"
        
        elif keep_start:
            # 只保留开头
            truncated_tokens = tokens[:available_tokens]
            return self.encoding.decode(truncated_tokens) + separator
        
        elif keep_end:
            # 只保留结尾
            truncated_tokens = tokens[-available_tokens:]
            return separator + self.encoding.decode(truncated_tokens)
        
        else:
            # 从中间截取
            start_pos = (len(tokens) - available_tokens) // 2
            truncated_tokens = tokens[start_pos:start_pos + available_tokens]
            return separator + self.encoding.decode(truncated_tokens) + separator
    
    def truncate_messages(self, messages: list, max_total_tokens: int) -> list:
        """
        截断消息列表，保持在token限制内
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            max_total_tokens: 最大总token数
            
        Returns:
            截断后的消息列表
        """
        if not messages:
            return []
        
        # 计算当前总token数
        total_tokens = sum(self.count_tokens(msg.get("content", "")) for msg in messages)
        
        if total_tokens <= max_total_tokens:
            return messages
        
        # 需要截断
        # 策略：保留system消息和最后几条消息
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # 计算system消息的token数
        system_tokens = sum(self.count_tokens(msg.get("content", "")) for msg in system_messages)
        
        # 为其他消息预留token
        available_tokens = max_total_tokens - system_tokens - 100  # 预留100 tokens作为缓冲
        
        # 从后往前保留消息
        kept_messages = []
        current_tokens = 0
        
        for msg in reversed(other_messages):
            msg_tokens = self.count_tokens(msg.get("content", ""))
            
            if current_tokens + msg_tokens <= available_tokens:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                # 如果这是第一条消息且太长，进行截断
                if not kept_messages:
                    remaining_tokens = available_tokens
                    truncated_content = self.truncate_text(
                        msg.get("content", ""),
                        remaining_tokens,
                        keep_start=True,
                        keep_end=True
                    )
                    msg_copy = msg.copy()
                    msg_copy["content"] = truncated_content
                    kept_messages.insert(0, msg_copy)
                break
        
        return system_messages + kept_messages


# 全局实例
_token_limiter_cache = {}


def get_token_limiter(model_name: str = "gpt-4") -> TokenLimiter:
    """
    获取TokenLimiter实例（带缓存）
    
    Args:
        model_name: 模型名称
        
    Returns:
        TokenLimiter实例
    """
    if model_name not in _token_limiter_cache:
        _token_limiter_cache[model_name] = TokenLimiter(model_name)
    return _token_limiter_cache[model_name]


def truncate_context(text: str, max_tokens: int = 25000, model_name: str = "gpt-4") -> str:
    """
    快速截断上下文文本
    
    Args:
        text: 输入文本
        max_tokens: 最大token数
        model_name: 模型名称
        
    Returns:
        截断后的文本
    """
    limiter = get_token_limiter(model_name)
    return limiter.truncate_text(text, max_tokens, keep_start=True, keep_end=True)



