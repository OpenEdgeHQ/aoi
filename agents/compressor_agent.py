# agents/compressor_agent.py

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio
import re

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig
from aworld.runner import Runners

from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager
from memory.memory_item import (
    AgentType, RawContextItem, CompressedContextItem,
    MemoryType, SubTaskItem
)
from prompts.loader import get_prompt_loader



class CompressorAgent(BaseAgent):
    """压缩器智能体 - 基于LLM的智能上下文压缩"""

    def __init__(self,
                 llm_config: AgentConfig,
                 memory_manager: MemoryManager,
                 window_size: int = 8000,
                 overlap_size: int = 2000,
                 min_compress_length: int = 500,
                 max_output_tokens: int = 5000,
                 max_context_tokens: int = 35000):
        """
        初始化压缩器智能体

        Args:
            llm_config: LLM配置
            memory_manager: 内存管理器
            window_size: 滑动窗口大小（tokens）
            overlap_size: 窗口重叠大小（tokens）
            min_compress_length: 最小压缩长度（字符）
            max_output_tokens: 目标压缩后的最大token数
            max_context_tokens: 模型最大输入token数
        """
        self.prompt_loader = get_prompt_loader()
        self.min_compress_length = min_compress_length
        self.max_output_tokens = max_output_tokens
        self.max_context_tokens = max_context_tokens

        super().__init__(
            name="Compressor Agent",
            agent_type=AgentType.COMPRESSOR,
            llm_config=llm_config,
            memory_manager=memory_manager,
            max_iterations=1
        )

        self.window_size = window_size
        self.overlap_size = overlap_size
        self.model_name = self._get_model_name()

    def _get_model_name(self) -> str:
        """安全地获取模型名称"""
        possible_attrs = ['model_name', 'model', 'llm_model', 'llm_model_name']
        for attr in possible_attrs:
            if hasattr(self.llm_config, attr):
                value = getattr(self.llm_config, attr)
                if value:
                    return value
        return "unknown_model"

    def _initialize_llm_agent(self) -> Agent:
        """初始化LLM智能体"""
        return Agent(
            name=self.name,
            conf=self.llm_config,
            system_prompt=self._get_system_prompt()
        )

    def _get_system_prompt(self) -> str:
        """获取系统提示词 - 智能压缩版本"""
        return self.prompt_loader.get_prompt(
            agent_type="compressor",
            prompt_type="system"
        )

    def _prepare_input(self,
                       instruction: str,
                       context: Dict[str, Any],
                       **kwargs) -> str:
        """
        准备LLM输入 - 实现抽象方法

        Args:
            instruction: 任务指令（压缩器可能不使用）
            context: 上下文信息
            **kwargs: 额外参数

        Returns:
            格式化的输入字符串
        """
        raw_output = kwargs.get("raw_output", context.get("raw_data", ""))
        source_agent = kwargs.get("source_agent", "unknown")

        # 检查输出长度
        output_length = len(raw_output)

        # 估算token数（粗略估算：1 token ≈ 4 字符）
        estimated_tokens = output_length // 4
        target_tokens = self.max_output_tokens

        # 使用prompt loader加载模板
        params = {
            "output_length": output_length,
            "raw_output": raw_output,
            "target_tokens": target_tokens,
            "estimated_tokens": estimated_tokens
        }

        return self.prompt_loader.get_prompt(
            agent_type="compressor",
            prompt_type="user",
            **params
        )

    def _process_decision(self,
                          decision: Dict[str, Any],
                          context: Dict[str, Any]) -> str:
        """
        处理决策结果 - 实现抽象方法

        Args:
            decision: LLM决策结果
            context: 上下文信息

        Returns:
            压缩后的输出字符串
        """
        output = decision.get("compressed_output", decision.get("raw_response", ""))

        if not output:
            # 如果没有输出，直接报错，不使用fallback
            raise ValueError("Compression failed: LLM returned empty output")

        # 记录到内存
        if self.memory_manager and output:
            self._store_compressed_context(output, context)

        self.agent_logger.info(f"Compression completed, output length: {len(output)}")
        return output

    def _get_fallback_decision(self) -> Dict[str, Any]:
        """
        获取备用决策 - 实现抽象方法
        注意：此函数不应被调用，因为我们直接抛出错误而不是使用fallback
        
        Returns:
            备用决策字典
        """
        raise NotImplementedError("Fallback decisions are not supported. Compression should either succeed or raise an error.")

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应 - 覆盖父类方法以适应压缩器特性

        Args:
            response: LLM原始响应

        Returns:
            解析后的决策字典
        """
        # 压缩器的响应直接作为压缩输出
        return {
            "compressed_output": response,
            "raw_response": response
        }

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数（粗略估算：1 token ≈ 4 字符）"""
        return len(text) // 4

    def _split_text_into_windows(self, text: str, window_size_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """
        将文本分割成多个滑动窗口
        
        Args:
            text: 输入文本
            window_size_tokens: 每个窗口的大小（tokens）
            overlap_tokens: 窗口之间的重叠大小（tokens）
            
        Returns:
            窗口文本列表
        """
        # 将文本按字符分割
        chars_per_token = 4  # 粗略估算
        window_size_chars = window_size_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token
        step_size = window_size_chars - overlap_chars
        
        if step_size <= 0:
            step_size = window_size_chars
        
        windows = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + window_size_chars, text_length)
            window_text = text[start:end]
            windows.append(window_text)
            
            # 如果已经到达文本末尾，退出
            if end >= text_length:
                break
                
            start += step_size
        
        self.agent_logger.info(f"  🪟 Split text into {len(windows)} windows (window_size={window_size_tokens} tokens, "
                        f"overlap={overlap_tokens} tokens)")
        
        return windows

    async def _sliding_window_compress(self, text: str, item: RawContextItem, 
                                      target_tokens_per_window: int) -> str:
        """
        使用无重复滑窗对超长文本进行压缩
        
        Args:
            text: 输入文本
            item: 原始上下文项
            target_tokens_per_window: 每个窗口压缩后的目标token数
            
        Returns:
            压缩后的文本
        """
        # 1. 将文本分割成窗口（窗口大小为max_context_tokens，无重叠）
        windows = self._split_text_into_windows(
            text, 
            window_size_tokens=self.max_context_tokens,
            overlap_tokens=0  # 无重叠滑窗
        )
        
        self.agent_logger.info(f"  🗜️  Compressing {len(windows)} windows with sliding window strategy...")
        
        # 2. 压缩每个窗口
        compressed_windows = []
        for idx, window_text in enumerate(windows, 1):
            self.agent_logger.info(f"    [{idx}/{len(windows)}] Compressing window {idx}...")
            
            # 为每个窗口调用LLM压缩，如果失败直接抛出异常
            compressed = await self._intelligent_compress_single(
                window_text, 
                item, 
                target_tokens=target_tokens_per_window
            )
            compressed_windows.append(compressed)
        
        # 3. 合并所有压缩后的窗口
        final_result = "\n\n".join([
            f"[Window {i+1}/{len(compressed_windows)}]\n{content}" 
            for i, content in enumerate(compressed_windows)
        ])
        
        return final_result


    async def _structure_raw_items_with_compression(self, raw_items: List[RawContextItem]) -> str:
        """
        将原始上下文项结构化为markdown格式，智能批量压缩Output部分
        
        新压缩逻辑：
        1. 遍历每个round结果
        2. 对round结果进行文本去重
        3. 如果去重后结果小于阈值threshold_per_round，直接保存
        4. 如果去重后结果大于阈值：
           - 如果小于模型最大输入阈值max_context_tokens，直接压缩，目标长度=max_output_tokens//n_round
           - 如果大于模型最大输入阈值max_context_tokens，使用滑窗压缩，目标长度=max_output_tokens//n_round//n_windows
        """
        if not raw_items:
            return "# No Raw Data Available\n"

        # 按时间排序
        sorted_items = sorted(raw_items, key=lambda x: (x.round_number, x.created_at))

        # 提取时间范围
        time_range = f"{sorted_items[0].created_at.strftime('%H:%M:%S')} - {sorted_items[-1].created_at.strftime('%H:%M:%S')}"

        # 计算每个round的阈值
        n_rounds = len(sorted_items)
        threshold_per_round = self.max_output_tokens // n_rounds if n_rounds > 0 else self.max_output_tokens
        
        self.agent_logger.info(f"📊 Compression strategy: {n_rounds} rounds")
        self.agent_logger.info(f"   - Threshold per round: {threshold_per_round} tokens")
        self.agent_logger.info(f"   - Max context tokens: {self.max_context_tokens} tokens")
        self.agent_logger.info(f"   - Max output tokens: {self.max_output_tokens} tokens")

        # 第一遍：预处理所有items，判断压缩策略
        items_data = []
        
        for idx, item in enumerate(sorted_items, 1):
            # 1. 处理输出并估算tokens（去重已在probe/executor中完成）
            output = self._process_output(item.raw_output)
            output_tokens = self._estimate_tokens(output)
            
            # 2. 判断压缩策略
            strategy = "keep"  # 默认策略：保持不变
            target_tokens = threshold_per_round
            
            if output_tokens <= threshold_per_round:
                # 小于阈值，直接保存
                strategy = "keep"
                self.agent_logger.info(f"  Round {item.round_number}: {output_tokens} tokens ≤ {threshold_per_round} → keep")
                
            elif output_tokens < self.max_context_tokens:
                # 大于阈值但小于模型输入限制，直接压缩
                strategy = "compress"
                target_tokens = self.max_output_tokens // n_rounds
                self.agent_logger.info(f"  Round {item.round_number}: {output_tokens} tokens → direct compress to {target_tokens} tokens")
                
            else:
                # 大于模型输入限制，使用滑窗压缩
                strategy = "sliding_window"
                # 计算需要多少个窗口
                n_windows = (output_tokens + self.max_context_tokens - 1) // self.max_context_tokens
                target_tokens = (self.max_output_tokens // n_rounds) // n_windows
                self.agent_logger.info(f"  Round {item.round_number}: {output_tokens} tokens → sliding window compress "
                               f"({n_windows} windows, target {target_tokens} tokens per window)")

            item_data = {
                'index': idx,
                'item': item,
                'output': output,
                'output_tokens': output_tokens,
                'strategy': strategy,
                'target_tokens': target_tokens
            }
            items_data.append(item_data)

        # 第二遍：根据策略处理每个round
        compressed_outputs = {}
        
        for item_data in items_data:
            idx = item_data['index']
            round_num = item_data['item'].round_number
            strategy = item_data['strategy']
            
            if strategy == "keep":
                # 直接保存原始输出
                compressed_outputs[idx] = item_data['output']
                
            elif strategy == "compress":
                # 直接压缩，失败直接抛出异常
                self.agent_logger.info(f"   🗜️  [{idx}/{n_rounds}] Round {round_num}: direct compression...")
                compressed = await self._intelligent_compress_single(
                    item_data['output'], 
                    item_data['item'],
                    target_tokens=item_data['target_tokens']
                )
                compressed_outputs[idx] = compressed
                self.agent_logger.info(f"   ✅ Round {round_num} compressed successfully")
                    
            elif strategy == "sliding_window":
                # 滑窗压缩，失败直接抛出异常
                self.agent_logger.info(f"   🪟 [{idx}/{n_rounds}] Round {round_num}: sliding window compression...")
                compressed = await self._sliding_window_compress(
                    item_data['output'],
                    item_data['item'],
                    target_tokens_per_window=item_data['target_tokens']
                )
                compressed_outputs[idx] = compressed
                self.agent_logger.info(f"   ✅ Round {round_num} sliding window compressed successfully")

        # 第三遍：构建最终输出 - 包含命令和压缩后的内容
        output_parts = []
        
        for item_data in items_data:
            idx = item_data['index']
            item = item_data['item']
            
            # 获取最终输出（压缩后的内容）
            final_output = compressed_outputs.get(idx, item_data['output'])
            
            # 如果有多个round，用分隔符分开
            if len(items_data) > 1:
                output_parts.append(f"[Round {item.round_number}]")
            
            # 添加命令信息（如果有）
            if hasattr(item, 'command') and item.command:
                output_parts.append(f"**Command**: `{item.command}`")
                output_parts.append("")  # 空行分隔
            
            # 添加结果标题和内容
            output_parts.append("**Result**:")
            output_parts.append(final_output)
            
            # 添加换行分隔
            if idx < len(items_data):
                output_parts.append("\n")

        # 返回包含命令和结果的完整内容
        return "\n".join(output_parts)

    async def _intelligent_compress_single(self, output_text: str, item: RawContextItem, 
                                          target_tokens: Optional[int] = None) -> str:
        """
        使用LLM智能压缩单个输出
        
        Args:
            output_text: 要压缩的文本
            item: 原始上下文项
            target_tokens: 目标token数（如果指定）
            
        Returns:
            压缩后的文本
            
        Raises:
            Exception: 如果压缩失败
        """
        # 1. 检查长度，决定是否需要压缩
        if len(output_text) < self.min_compress_length:
            self.agent_logger.info(f"Text length ({len(output_text)}) below threshold ({self.min_compress_length}), returning as-is")
            return output_text
        
        # 2. 文本长度超过阈值，使用LLM压缩
        self.agent_logger.info(f"Text length ({len(output_text)}) exceeds threshold ({self.min_compress_length}), compressing with LLM")
        
        # 通过process方法调用LLM（使用BaseAgent的框架）
        # 如果压缩失败，会直接抛出异常
        compressed = await self.process(
            task_instruction="",  # 压缩器不需要instruction
            context={"raw_data": output_text, "target_tokens": target_tokens},
            raw_output=output_text,
            source_agent=item.source_agent.value if item.source_agent else "unknown"
        )
        return compressed

    def _process_output(self, output: Any) -> str:
        """
        处理输出数据，支持字典、列表和字符串

        Args:
            output: 原始输出

        Returns:
            处理后的字符串
        """
        if isinstance(output, dict):
            try:
                return json.dumps(output, indent=2, ensure_ascii=False)
            except:
                return str(output)
        elif isinstance(output, list):
            try:
                return json.dumps(output, indent=2, ensure_ascii=False)
            except:
                return str(output)
        elif isinstance(output, str):
            return output
        else:
            return str(output)

    def _store_compressed_context(self, output: str, context: Dict[str, Any]):
        """存储压缩上下文到内存"""
        # 简化：不再进行硬编码分析，直接存储
        key_findings = []

        # 计算压缩率
        original_size = len(context.get("raw_data", ""))
        compressed_size = len(output)
        compression_ratio = 0.0

        if original_size > 0 and compressed_size < original_size:
            compression_ratio = 1.0 - (compressed_size / original_size)

        compressed_item = CompressedContextItem(
            source_items=context.get("source_ids", []),
            compression_ratio=compression_ratio,
            original_size=original_size,
            compressed_size=compressed_size,
            summary=f"Compressed from {original_size} to {compressed_size} chars",
            key_findings=key_findings,
            anomaly_indicators={},
            timeline=self._extract_timeline(context),
            recommendations=self._generate_recommendations(),
            semantic_tags=[],  # 让LLM自己提取semantic信息
            confidence_score=0.95,
            compression_model=self.model_name,
            compression_prompt="Enhanced error-preserving compression",
            metadata={
                "session_id": context.get("session_id", ""),
                "compression_timestamp": datetime.now().isoformat()
            }
        )

        self.memory_manager.add_item(compressed_item, self.agent_type)
        return compressed_item.id

    def _extract_timeline(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取时间线信息"""
        timeline = []
        raw_items = context.get("raw_items", [])

        for item in raw_items:
            if isinstance(item, RawContextItem):
                timeline.append({
                    "timestamp": item.created_at.isoformat(),
                    "round": item.round_number,
                    "event": f"{item.source_agent.value if item.source_agent else 'unknown'}: {item.command[:50] if item.command else 'N/A'}",
                    "success": getattr(item, 'success', True)
                })

        return sorted(timeline, key=lambda x: x["timestamp"])

    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        return ["Review compressed output for issues"]


    async def compress_context(self,
                               raw_data: str = "",
                               source_agent: str = "",
                               round_info: str = "",
                               source_ids: List[str] = None,
                               session_id: str = None,
                               raw_items: List[RawContextItem] = None) -> str:
        """
        压缩上下文 - 主要接口


        Raises:
            Exception: 如果压缩失败
        """
        # 如果提供了raw_items，使用新的结构化处理
        if raw_items:
            result = await self._structure_raw_items_with_compression(raw_items)
            source_ids = [item.id for item in raw_items]

        # 如果没有原始数据，从内存获取
        elif not raw_data and source_ids and self.memory_manager:
            raw_items = []
            for item_id in source_ids:
                item = self.memory_manager.get_item(item_id, self.agent_type)
                if isinstance(item, RawContextItem):
                    raw_items.append(item)

            if raw_items:
                result = await self._structure_raw_items_with_compression(raw_items)
            else:
                result = "[NO DATA TO COMPRESS]"

        # 如果有原始数据但没有raw_items
        elif raw_data:
            # 检查长度决定是否压缩
            if len(raw_data) < self.min_compress_length:
                result = raw_data
            else:
                # 直接报错，不再使用fallback
                raise ValueError(f"Raw data length ({len(raw_data)}) exceeds threshold but raw_items not provided. Cannot compress without RawContextItem.")
        else:
            result = "[NO DATA TO COMPRESS]"

        # 存储压缩结果到内存
        if self.memory_manager and result != "[NO DATA TO COMPRESS]":
            context = {
                "raw_data": raw_data or "",
                "source_ids": source_ids or [],
                "session_id": session_id,
                "raw_items": raw_items or []
            }
            self._store_compressed_context(result, context)

        return result

    async def compressor_run(self, session_id: str, current_subtask: Optional[SubTaskItem] = None) -> str:
        """
        主要接口函数 - 根据session_id获取并压缩数据
        
        Args:
            session_id: 会话ID
            current_subtask: 当前子任务，用于获取 iteration 信息
        """
        try:
            # 从 current_subtask 获取 iteration
            current_iteration = current_subtask.iteration_number if current_subtask else None
            
            self.agent_logger.info(f"🗜️ Starting compression for session: {session_id}")

            if not self.memory_manager:
                return "[ERROR: No memory manager available]"

            # 查询该session的原始上下文（只查询当前iteration的数据，避免重复压缩）
            # RAW_CONTEXT 是由 Probe 和 Executor 产生的，需要查询这两个 agent_type
            if current_iteration is not None:
                # 只查询当前 iteration 的数据
                filters = {
                    "metadata": lambda x: (
                        isinstance(x, dict) and 
                        x.get("session_id") == session_id and 
                        x.get("iteration") == current_iteration
                    )
                }
                self.agent_logger.info(f"🔍 Querying RAW_CONTEXT for iteration {current_iteration} only")
            else:
                # 查询所有数据（用于特殊情况，但通常应该有 current_subtask）
                filters = {
                    "metadata": lambda x: x.get("session_id") == session_id if isinstance(x, dict) else False
                }
                self.agent_logger.info(f"🔍 Querying all RAW_CONTEXT for session")

            raw_items = []
            
            # 查询 RAW_CONTEXT (使用 COMPRESSOR 作为查询者，因为它有READ权限)
            # 然后通过 filters 和 source_agent 属性来过滤数据
            all_raw_items = self.memory_manager.query_items(
                agent_type=AgentType.COMPRESSOR,  # 查询者（有READ权限）
                memory_type=MemoryType.RAW_CONTEXT,
                filters=filters,
                limit=1000,
                sort_by="created_at",
                descending=False
            )
            self.agent_logger.info(f"📊 Query returned {len(all_raw_items)} total RAW_CONTEXT items")
            
            # 手动过滤出 Probe 和 Executor 的数据
            for item in all_raw_items:
                if hasattr(item, 'source_agent') and item.source_agent in [AgentType.PROBE, AgentType.EXECUTOR]:
                    raw_items.append(item)
            
            self.agent_logger.info(f"📊 Filtered to {len(raw_items)} items from Probe/Executor")
            
            # 按 created_at 排序
            raw_items.sort(key=lambda x: x.created_at if hasattr(x, 'created_at') else 0)

            if not raw_items:
                self.agent_logger.warning(f"⚠️ No data found for session: {session_id}")
                return f"[NO DATA FOUND FOR SESSION: {session_id}]"

            # 添加压缩前统计日志
            total_size = sum(len(str(item.raw_output)) for item in raw_items)
            self.agent_logger.info(f"📊 Found {len(raw_items)} items, total size: {total_size} chars")

            # 执行智能压缩
            compressed_result = await self.compress_context(
                raw_data="",
                source_agent="compressor_run",
                round_info=f"Batch compression for session {session_id}",
                source_ids=None,
                session_id=session_id,
                raw_items=raw_items
            )

            # 添加压缩后日志
            compressed_size = len(compressed_result)
            compression_ratio = 1.0 - (compressed_size / total_size) if total_size > 0 else 0
            self.agent_logger.info(f"✅ Compression completed: {compressed_size} chars (ratio: {compression_ratio:.2%})")

            return compressed_result

        except Exception as e:
            self.logger.error(f"Error in compressor_run: {e}", exc_info=True)
            return f"[ERROR: {str(e)}]"
