# memory/memory_manager.py
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from threading import Lock
from enum import Enum

from memory.memory_item import (
    BaseMemoryItem, RawContextItem, SubTaskItem, CompressedContextItem,
    MemoryType, TaskStatus, AgentType
)


class AccessPermission(Enum):
    """访问权限枚举"""
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    NONE = "none"


class MemoryManager:
    """
    内存管理器 - 实现三层数据存储架构
    支持原始上下文、子任务队列和压缩上下文的统一管理
    """

    def __init__(self,
                 max_raw_items: int = 100000,
                 max_compressed_items: int = 10000,
                 max_task_items: int = 2000,
                 enable_auto_cleanup: bool = True):
        """
        初始化内存管理器

        Args:
            max_raw_items: 原始上下文最大存储数量
            max_compressed_items: 压缩上下文最大存储数量
            max_task_items: 子任务最大存储数量
            enable_auto_cleanup: 是否启用自动清理
        """
        # 三层存储结构
        self.raw_context_store: Dict[str, RawContextItem] = {}
        self.sub_task_store: Dict[str, SubTaskItem] = {}
        self.compressed_context_store: Dict[str, CompressedContextItem] = {}

        # 索引结构 - 支持快速查询
        self.agent_index: Dict[AgentType, List[str]] = defaultdict(list)
        self.task_status_index: Dict[TaskStatus, List[str]] = defaultdict(list)
        self.timestamp_index: Dict[str, datetime] = {}
        self.session_index: Dict[str, List[str]] = defaultdict(list)

        # 访问控制
        self.access_rules = self._initialize_access_rules()

        # 配置
        self.max_raw_items = max_raw_items
        self.max_compressed_items = max_compressed_items
        self.max_task_items = max_task_items
        self.enable_auto_cleanup = enable_auto_cleanup

        # 统计信息
        self.stats = {
            "total_writes": 0,
            "total_reads": 0,
            "total_cleanups": 0,
            "total_compressions": 0
        }

        # 线程安全
        self.lock = Lock()

        # 内存快照（简化版，不使用MemorySnapshot类）
        self.memory_snapshots: deque = deque(maxlen=10)

        # 日志
        self.logger = logging.getLogger(__name__)

    def _initialize_access_rules(self) -> Dict[tuple, AccessPermission]:
        """初始化访问控制规则"""
        rules = {}

        # 原始上下文访问规则
        rules[(AgentType.PROBE, MemoryType.RAW_CONTEXT)] = AccessPermission.WRITE
        rules[(AgentType.EXECUTOR, MemoryType.RAW_CONTEXT)] = AccessPermission.WRITE
        rules[(AgentType.COMPRESSOR, MemoryType.RAW_CONTEXT)] = AccessPermission.READ
        rules[(AgentType.OBSERVER, MemoryType.RAW_CONTEXT)] = AccessPermission.NONE

        # 子任务访问规则
        rules[(AgentType.OBSERVER, MemoryType.SUB_TASK)] = AccessPermission.READ_WRITE
        rules[(AgentType.PROBE, MemoryType.SUB_TASK)] = AccessPermission.READ
        rules[(AgentType.EXECUTOR, MemoryType.SUB_TASK)] = AccessPermission.READ
        rules[(AgentType.COMPRESSOR, MemoryType.SUB_TASK)] = AccessPermission.NONE

        # 压缩上下文访问规则
        rules[(AgentType.OBSERVER, MemoryType.COMPRESSED_CONTEXT)] = AccessPermission.READ
        rules[(AgentType.COMPRESSOR, MemoryType.COMPRESSED_CONTEXT)] = AccessPermission.WRITE
        rules[(AgentType.PROBE, MemoryType.COMPRESSED_CONTEXT)] = AccessPermission.NONE
        rules[(AgentType.EXECUTOR, MemoryType.COMPRESSED_CONTEXT)] = AccessPermission.NONE

        return rules

    def check_permission(self, agent_type: AgentType, memory_type: MemoryType,
                         operation: str) -> bool:
        """检查访问权限"""
        permission = self.access_rules.get((agent_type, memory_type), AccessPermission.NONE)

        if operation == "read":
            return permission in [AccessPermission.READ, AccessPermission.READ_WRITE]
        elif operation == "write":
            return permission in [AccessPermission.WRITE, AccessPermission.READ_WRITE]
        return False

    def add_item(self, item: BaseMemoryItem, agent_type: AgentType) -> bool:
        """
        添加内存项

        Args:
            item: 内存项
            agent_type: 智能体类型

        Returns:
            是否添加成功
        """
        with self.lock:
            # 权限检查
            if not self.check_permission(agent_type, item.memory_type, "write"):
                self.logger.warning(f"Permission denied: {agent_type} cannot write {item.memory_type}")
                return False

            # 根据类型存储
            if isinstance(item, RawContextItem):
                self._add_raw_context(item)
            elif isinstance(item, SubTaskItem):
                self._add_sub_task(item)
            elif isinstance(item, CompressedContextItem):
                self._add_compressed_context(item)
            else:
                self.logger.error(f"Unknown item type: {type(item)}")
                return False

            # 更新索引
            self._update_indices(item, agent_type)

            self.stats["total_writes"] += 1

            # 自动清理
            if self.enable_auto_cleanup:
                self._auto_cleanup()

            return True

    def _add_raw_context(self, item: RawContextItem):
        """添加原始上下文"""
        self.raw_context_store[item.id] = item
        self.agent_index[item.source_agent].append(item.id)
        self.timestamp_index[item.id] = item.created_at

        # 添加到session索引
        if item.metadata and 'session_id' in item.metadata:
            self.session_index[item.metadata['session_id']].append(item.id)

        # 容量控制
        if len(self.raw_context_store) > self.max_raw_items:
            self._evict_oldest_raw_context()

    def _add_sub_task(self, item: SubTaskItem):
        """添加子任务"""
        self.sub_task_store[item.id] = item
        self.task_status_index[item.status].append(item.id)
        self.timestamp_index[item.id] = item.created_at

        # 容量控制
        if len(self.sub_task_store) > self.max_task_items:
            self._archive_completed_tasks()

    def _add_compressed_context(self, item: CompressedContextItem):
        """添加压缩上下文"""
        self.compressed_context_store[item.id] = item
        self.timestamp_index[item.id] = item.created_at
        self.stats["total_compressions"] += 1

        # 添加到session索引
        if item.metadata and 'session_id' in item.metadata:
            self.session_index[item.metadata['session_id']].append(item.id)

        # 容量控制
        if len(self.compressed_context_store) > self.max_compressed_items:
            self._evict_oldest_compressed_context()

    def get_item(self, item_id: str, agent_type: AgentType) -> Optional[BaseMemoryItem]:
        """获取内存项"""
        with self.lock:
            # 查找项目
            item = None
            if item_id in self.raw_context_store:
                item = self.raw_context_store[item_id]
            elif item_id in self.sub_task_store:
                item = self.sub_task_store[item_id]
            elif item_id in self.compressed_context_store:
                item = self.compressed_context_store[item_id]

            if item and self.check_permission(agent_type, item.memory_type, "read"):
                self.stats["total_reads"] += 1
                return item

            return None

    def query_items(self,
                    agent_type: AgentType,
                    memory_type: Optional[MemoryType] = None,
                    filters: Optional[Dict[str, Any]] = None,
                    limit: int = 100,
                    sort_by: str = "created_at",
                    descending: bool = True) -> List[BaseMemoryItem]:
        """
        查询内存项

        Args:
            agent_type: 查询的智能体类型
            memory_type: 内存类型过滤
            filters: 额外过滤条件
            limit: 返回数量限制
            sort_by: 排序字段
            descending: 是否降序

        Returns:
            符合条件的内存项列表
        """
        with self.lock:
            results = []

            # 确定查询范围
            if memory_type == MemoryType.RAW_CONTEXT:
                if self.check_permission(agent_type, MemoryType.RAW_CONTEXT, "read"):
                    results = list(self.raw_context_store.values())
            elif memory_type == MemoryType.SUB_TASK:
                if self.check_permission(agent_type, MemoryType.SUB_TASK, "read"):
                    results = list(self.sub_task_store.values())
            elif memory_type == MemoryType.COMPRESSED_CONTEXT:
                if self.check_permission(agent_type, MemoryType.COMPRESSED_CONTEXT, "read"):
                    results = list(self.compressed_context_store.values())
            else:
                # 查询所有有权限的类型
                for mt in MemoryType:
                    if self.check_permission(agent_type, mt, "read"):
                        if mt == MemoryType.RAW_CONTEXT:
                            results.extend(self.raw_context_store.values())
                        elif mt == MemoryType.SUB_TASK:
                            results.extend(self.sub_task_store.values())
                        elif mt == MemoryType.COMPRESSED_CONTEXT:
                            results.extend(self.compressed_context_store.values())

            # 应用过滤器
            if filters:
                results = self._apply_filters(results, filters)

            # 排序
            if hasattr(results[0] if results else None, sort_by):
                results.sort(key=lambda x: getattr(x, sort_by), reverse=descending)

            # 限制数量
            results = results[:limit]

            self.stats["total_reads"] += len(results)
            return results

    def _apply_filters(self, items: List[BaseMemoryItem],
                       filters: Dict[str, Any]) -> List[BaseMemoryItem]:
        """应用过滤条件"""
        filtered = []
        for item in items:
            match = True
            for key, value in filters.items():
                if hasattr(item, key):
                    item_value = getattr(item, key)
                    if callable(value):
                        # 支持函数过滤器
                        if not value(item_value):
                            match = False
                            break
                    elif item_value != value:
                        match = False
                        break
            if match:
                filtered.append(item)
        return filtered

    def update_item(self, item: BaseMemoryItem, agent_type: AgentType) -> bool:
        """
        更新内存项

        Args:
            item: 要更新的内存项
            agent_type: 相关的智能体类型

        Returns:
            是否更新成功
        """
        with self.lock:
            # 权限检查
            if not self.check_permission(agent_type, item.memory_type, "write"):
                self.logger.warning(f"Permission denied: {agent_type} cannot update {item.memory_type}")
                return False

            # 根据内存类型获取存储
            if isinstance(item, RawContextItem):
                if item.id in self.raw_context_store:
                    item.update()  # 更新时间戳
                    self.raw_context_store[item.id] = item
                    return True
            elif isinstance(item, SubTaskItem):
                if item.id in self.sub_task_store:
                    # 更新任务状态索引
                    old_task = self.sub_task_store.get(item.id)
                    if old_task and old_task.status != item.status:
                        # 从旧状态索引中移除
                        if item.id in self.task_status_index[old_task.status]:
                            self.task_status_index[old_task.status].remove(item.id)
                        # 添加到新状态索引
                        self.task_status_index[item.status].append(item.id)

                    item.update()  # 更新时间戳
                    self.sub_task_store[item.id] = item
                    return True
            elif isinstance(item, CompressedContextItem):
                if item.id in self.compressed_context_store:
                    item.update()  # 更新时间戳
                    self.compressed_context_store[item.id] = item
                    return True

            # 如果没找到，尝试添加为新项
            self.logger.warning(f"Item {item.id} not found, adding as new")
            return self.add_item(item, agent_type)

    def _update_indices(self, item: BaseMemoryItem, agent_type: AgentType):
        """
        更新索引

        Args:
            item: 内存项
            agent_type: 智能体类型
        """
        # 更新agent_type索引（已在各个_add_*方法中处理）

        # 更新session_id索引（如果有）
        if hasattr(item, 'metadata') and item.metadata:
            session_id = item.metadata.get('session_id')
            if session_id and item.id not in self.session_index[session_id]:
                self.session_index[session_id].append(item.id)

    def get_pending_tasks(self, target_agent: AgentType) -> List[SubTaskItem]:
        """获取特定智能体的待执行任务"""
        with self.lock:
            pending_tasks = []
            for task_id in self.task_status_index[TaskStatus.PENDING]:
                task = self.sub_task_store.get(task_id)
                if task and task.target_agent == target_agent:
                    # 检查任务是否可执行
                    if task.is_executable():
                        pending_tasks.append(task)

            # 按优先级和迭代号排序
            pending_tasks.sort(key=lambda x: (x.iteration_number, -x.priority))
            return pending_tasks

    def get_raw_contexts_for_compression(self, limit: int = 10) -> List[RawContextItem]:
        """获取需要压缩的原始上下文"""
        with self.lock:
            # 获取最近未压缩的原始上下文
            candidates = []
            for item in self.raw_context_store.values():
                if not item.metadata.get("compressed", False):
                    candidates.append(item)

            # 按时间排序，优先压缩旧数据
            candidates.sort(key=lambda x: x.created_at)
            return candidates[:limit]

    def mark_as_compressed(self, item_ids: List[str]):
        """标记原始上下文已压缩"""
        with self.lock:
            for item_id in item_ids:
                if item_id in self.raw_context_store:
                    self.raw_context_store[item_id].metadata["compressed"] = True
                    self.raw_context_store[item_id].update()

    def create_snapshot(self, description: str = "") -> Dict[str, Any]:
        """
        创建内存快照（简化版）

        Returns:
            快照数据字典
        """
        with self.lock:
            snapshot_data = {
                "snapshot_id": str(datetime.now().timestamp()),
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "data": {
                    "raw_context_count": len(self.raw_context_store),
                    "sub_task_count": len(self.sub_task_store),
                    "compressed_context_count": len(self.compressed_context_store),
                    "raw_context_ids": list(self.raw_context_store.keys()),
                    "sub_task_ids": list(self.sub_task_store.keys()),
                    "compressed_context_ids": list(self.compressed_context_store.keys()),
                },
                "stats": self.stats.copy(),
                "task_status": {
                    status.value: len(ids)
                    for status, ids in self.task_status_index.items()
                }
            }

            # 保存快照
            self.memory_snapshots.append(snapshot_data)

            return snapshot_data

    def restore_from_snapshot(self, snapshot_id: str) -> bool:
        """
        从快照恢复（仅恢复索引，不恢复实际数据）

        Args:
            snapshot_id: 快照ID

        Returns:
            是否恢复成功
        """
        with self.lock:
            for snapshot in self.memory_snapshots:
                if snapshot.get("snapshot_id") == snapshot_id:
                    # 仅恢复统计信息
                    self.stats = snapshot.get("stats", self.stats).copy()
                    self.logger.info(f"Restored stats from snapshot {snapshot_id}")
                    return True
            return False

    def _rebuild_indexes(self):
        """重建所有索引"""
        self.agent_index.clear()
        self.task_status_index.clear()
        self.timestamp_index.clear()
        self.session_index.clear()

        # 重建原始上下文索引
        for item in self.raw_context_store.values():
            self.agent_index[item.source_agent].append(item.id)
            self.timestamp_index[item.id] = item.created_at
            if item.metadata and 'session_id' in item.metadata:
                self.session_index[item.metadata['session_id']].append(item.id)

        # 重建任务索引
        for item in self.sub_task_store.values():
            self.task_status_index[item.status].append(item.id)
            self.timestamp_index[item.id] = item.created_at

        # 重建压缩上下文索引
        for item in self.compressed_context_store.values():
            self.timestamp_index[item.id] = item.created_at
            if item.metadata and 'session_id' in item.metadata:
                self.session_index[item.metadata['session_id']].append(item.id)

    def _auto_cleanup(self):
        """自动清理过期数据"""
        now = datetime.now()

        # 清理超过24小时的原始上下文
        one_day_ago = now - timedelta(hours=24)
        expired_raw = [
            item_id for item_id, timestamp in self.timestamp_index.items()
            if item_id in self.raw_context_store and timestamp < one_day_ago
        ]

        for item_id in expired_raw:
            if item_id in self.raw_context_store:
                del self.raw_context_store[item_id]
                del self.timestamp_index[item_id]

        # 清理超过7天的压缩上下文
        seven_days_ago = now - timedelta(days=7)
        expired_compressed = [
            item_id for item_id, timestamp in self.timestamp_index.items()
            if item_id in self.compressed_context_store and timestamp < seven_days_ago
        ]

        for item_id in expired_compressed:
            if item_id in self.compressed_context_store:
                del self.compressed_context_store[item_id]
                del self.timestamp_index[item_id]

        # 归档已完成的任务
        self._archive_completed_tasks()

        if expired_raw or expired_compressed:
            self.stats["total_cleanups"] += 1
            self._rebuild_indexes()

    def _archive_completed_tasks(self):
        """归档已完成的任务"""
        completed_ids = list(self.task_status_index[TaskStatus.COMPLETED])
        failed_ids = list(self.task_status_index[TaskStatus.FAILED])
        skipped_ids = list(self.task_status_index[TaskStatus.SKIPPED])

        # 归档超过1小时的已完成/失败/跳过任务
        one_hour_ago = datetime.now() - timedelta(hours=1)

        for task_id in completed_ids + failed_ids + skipped_ids:
            task = self.sub_task_store.get(task_id)
            if task and task.completion_time and task.completion_time < one_hour_ago:
                # 这里可以选择将任务移到持久化存储
                # 目前只是标记
                task.metadata["archived"] = True

    def _evict_oldest_raw_context(self):
        """驱逐最旧的原始上下文"""
        if not self.raw_context_store:
            return

        # 找到最旧的项
        oldest_id = None
        oldest_time = datetime.now()

        for item_id, item in self.raw_context_store.items():
            # 跳过关键信息
            if item.metadata.get("critical", False):
                continue

            if item.created_at < oldest_time:
                oldest_id = item_id
                oldest_time = item.created_at

        if oldest_id:
            del self.raw_context_store[oldest_id]
            # 清理相关索引
            for agent, ids in self.agent_index.items():
                if oldest_id in ids:
                    ids.remove(oldest_id)
            if oldest_id in self.timestamp_index:
                del self.timestamp_index[oldest_id]

    def _evict_oldest_compressed_context(self):
        """驱逐最旧的压缩上下文"""
        if not self.compressed_context_store:
            return

        # 找到最旧的低置信度项
        oldest_id = None
        oldest_time = datetime.now()

        for item_id, item in self.compressed_context_store.items():
            if item.confidence_score < 0.5 and item.created_at < oldest_time:
                oldest_id = item_id
                oldest_time = item.created_at

        if oldest_id:
            del self.compressed_context_store[oldest_id]
            if oldest_id in self.timestamp_index:
                del self.timestamp_index[oldest_id]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                **self.stats,
                "current_items": {
                    "raw_context": len(self.raw_context_store),
                    "sub_tasks": len(self.sub_task_store),
                    "compressed_context": len(self.compressed_context_store)
                },
                "task_status": {
                    status.value: len(ids)
                    for status, ids in self.task_status_index.items()
                },
                "snapshots": len(self.memory_snapshots),
                "sessions": len(self.session_index)
            }

    def clear_all(self, agent_type: AgentType = AgentType.OBSERVER):
        """清空所有内存（需要Observer权限）"""
        if agent_type != AgentType.OBSERVER:
            self.logger.warning("Only Observer can clear all memory")
            return

        with self.lock:
            self.raw_context_store.clear()
            self.sub_task_store.clear()
            self.compressed_context_store.clear()
            self.agent_index.clear()
            self.task_status_index.clear()
            self.timestamp_index.clear()
            self.session_index.clear()
            self.memory_snapshots.clear()

            # 重置统计
            self.stats = {
                "total_writes": 0,
                "total_reads": 0,
                "total_cleanups": 0,
                "total_compressions": 0
            }

            self.logger.info("All memory cleared")

    def get_items_by_session(self, session_id: str) -> Dict[str, List[BaseMemoryItem]]:
        """
        根据session_id获取所有相关的内存项

        Args:
            session_id: 会话ID

        Returns:
            按类型分组的内存项
        """
        with self.lock:
            result = {
                "raw_context": [],
                "compressed_context": [],
                "sub_tasks": []
            }

            # 从session索引获取ID列表
            item_ids = self.session_index.get(session_id, [])

            for item_id in item_ids:
                if item_id in self.raw_context_store:
                    result["raw_context"].append(self.raw_context_store[item_id])
                elif item_id in self.compressed_context_store:
                    result["compressed_context"].append(self.compressed_context_store[item_id])

            # 子任务通过遍历查找（因为可能在task_context中）
            for task in self.sub_task_store.values():
                if task.task_context and task.task_context.get("session_id") == session_id:
                    result["sub_tasks"].append(task)

            return result