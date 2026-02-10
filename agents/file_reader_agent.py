# agents/file_reader_agent.py
import os
import re
import json
import subprocess
from typing import List, Optional, Dict, Any
from datetime import datetime

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig
from aworld.runner import Runners
from utils.logger_config import AgentLogger
from prompts.loader import get_prompt_loader


class FileReaderAgent:
    """
    文件读取Agent - 使用LLM智能选择并读取最相关的文件
    """
    
    def __init__(self, llm_config: AgentConfig):
        """
        初始化文件读取Agent
        
        Args:
            llm_config: LLM配置
        """
        self.llm_config = llm_config
        self.agent_logger = AgentLogger("FILE_READER")
        self.prompt_loader = get_prompt_loader()
        
        # 初始化LLM agent (用于文件选择)
        self.llm_agent = Agent(
            name="File Reader Agent",
            conf=llm_config,
            system_prompt=self._get_system_prompt()
        )
        
        # 初始化路径提取Agent
        self.path_extraction_agent = Agent(
            name="Path Extraction Agent",
            conf=llm_config,
            system_prompt=self._get_path_extraction_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.prompt_loader.get_prompt(
            agent_type="file_reader",
            prompt_type="system",
            max_files=3
        )

    def _get_path_extraction_system_prompt(self) -> str:
        """获取路径提取系统提示词"""
        return self.prompt_loader.get_prompt(
            agent_type="file_reader",
            prompt_type="path_extraction_system"
        )
    
    async def detect_csv_paths(self, result_text: str, task_instruction: str = "") -> List[Dict[str, str]]:
        """
        使用LLM从结果文本中检测CSV/log/txt文件路径
        
        Args:
            result_text: 命令执行结果文本
            task_instruction: 任务指引（帮助LLM识别相关文件）
            
        Returns:
            检测到的文件信息列表 [{"path": "...", "name": "...", "dir": "..."}]
        """
        try:
            # 构建用户prompt
            user_prompt = self.prompt_loader.get_prompt(
                agent_type="file_reader",
                prompt_type="path_extraction_user",
                task_instruction=task_instruction or "Analyze the metrics/traces/logs data",
                result_text=result_text
            )

            # 调用LLM提取路径
            self.agent_logger.info("🔍 Using LLM to extract file paths from command output...")
            llm_response = await Runners.run(
                input=user_prompt,
                agent=self.path_extraction_agent
            )
            
            # 提取响应文本
            response_text = llm_response.answer if hasattr(llm_response, 'answer') else str(llm_response)
            
            # 解析JSON响应
            extraction_result = self._parse_llm_response(response_text)
            detected_files = extraction_result.get("detected_files", [])
            reasoning = extraction_result.get("reasoning", "")
            
            # 转换为标准格式
            file_infos = []
            seen = set()
            
            for file_path in detected_files:
                if file_path and file_path not in seen:
                    seen.add(file_path)
                    file_infos.append({
                        "path": file_path,
                        "name": os.path.basename(file_path),
                        "dir": os.path.dirname(file_path)
                    })
            
            if file_infos:
                self.agent_logger.info(f"✅ Detected {len(file_infos)} file(s)")
                self.agent_logger.info(f"💡 Reasoning: {reasoning}")
            else:
                self.agent_logger.warning("⚠️ No files detected by LLM")
            
            return file_infos
            
        except Exception as e:
            self.agent_logger.error(f"❌ Error in LLM path extraction: {e}")
            # 降级：使用正则表达式备用方案
            self.agent_logger.warning("⚠️ Falling back to regex pattern matching")
            return self._fallback_detect_paths(result_text)
    
    def _fallback_detect_paths(self, result_text: str) -> List[Dict[str, str]]:
        """
        备用方案：使用正则表达式检测文件路径
        
        Args:
            result_text: 命令执行结果文本
            
        Returns:
            检测到的文件信息列表
        """
        # 匹配绝对路径中的CSV/log/txt文件
        path_patterns = [
            r'/[^\s\n]+\.(?:csv|log|txt)',  # Unix路径
            r'[A-Z]:\\[^\s\n]+\.(?:csv|log|txt)',  # Windows路径
        ]
        
        file_paths = []
        seen = set()
        
        for pattern in path_patterns:
            matches = re.findall(pattern, result_text)
            for path in matches:
                if path not in seen:
                    seen.add(path)
                    file_paths.append({
                        "path": path,
                        "name": os.path.basename(path),
                        "dir": os.path.dirname(path)
                    })
        
        if file_paths:
            self.agent_logger.info(f"📁 Regex detected {len(file_paths)} file(s)")
        
        return file_paths
    
    def _format_file_list(self, file_infos: List[Dict[str, str]]) -> str:
        """格式化文件列表用于LLM"""
        formatted = []
        for i, info in enumerate(file_infos, 1):
            formatted.append(f"{i}. {info['path']}")
            formatted.append(f"   - Filename: {info['name']}")
            formatted.append(f"   - Directory: {info['dir']}")
        
        return "\n".join(formatted)
    
    async def select_files(
        self,
        file_infos: List[Dict[str, str]],
        task_instruction: str,
        max_files: int = 2
    ) -> List[str]:
        """
        使用LLM选择最相关的文件
        
        Args:
            file_infos: 检测到的文件信息列表
            task_instruction: 当前任务指引
            max_files: 最多选择文件数

        Returns:
            选择的文件路径列表
        """
        if not file_infos:
            return []
        
        if len(file_infos) <= max_files:
            # 如果文件数不超过限制，全部返回
            return [info['path'] for info in file_infos]
        
        # 构建用户prompt
        user_prompt = self.prompt_loader.get_prompt(
            agent_type="file_reader",
            prompt_type="user",
            task_instruction=task_instruction,
            file_list=self._format_file_list(file_infos),
            max_files=max_files
        )
        
        try:
            # 调用LLM选择文件
            self.agent_logger.info(f"🤔 Using LLM to select {max_files} most relevant files from {len(file_infos)} options...")
            llm_response = await Runners.run(
                input=user_prompt,
                agent=self.llm_agent
            )
            
            # 提取响应文本
            response_text = llm_response.answer if hasattr(llm_response, 'answer') else str(llm_response)
            
            # 解析JSON响应
            decision = self._parse_llm_response(response_text)
            selected_files = decision.get("selected_files", [])
            reasoning = decision.get("reasoning", "")
            
            if selected_files:
                self.agent_logger.info(f"✅ Selected {len(selected_files)} files")
                self.agent_logger.info(f"💡 Reasoning: {reasoning[:100]}...")
                return selected_files
            else:
                # 如果LLM没有选择，返回前max_files个
                self.agent_logger.warning("⚠️ LLM didn't select files, using first N files")
                return [info['path'] for info in file_infos[:max_files]]
        
        except Exception as e:
            self.agent_logger.error(f"❌ Error in LLM file selection: {e}")
            # 降级：返回前max_files个
            return [info['path'] for info in file_infos[:max_files]]
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.agent_logger.error(f"Failed to parse LLM response: {e}")
        
        return {"selected_files": [], "reasoning": ""}
    
    def read_file(self, file_path: str, max_lines: int = 100) -> str:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            max_lines: 最大读取行数
            
        Returns:
            文件内容
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            # 使用head命令读取前N行（更高效）
            if os.name == 'posix':  # Unix/Linux/Mac
                cmd = f"head -n {max_lines} '{file_path}'"
            else:  # Windows
                cmd = f"powershell -Command \"Get-Content '{file_path}' -TotalCount {max_lines}\""
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                content = result.stdout
                # 添加文件信息头
                header = f"\n{'='*60}\n=== File: {os.path.basename(file_path)} (first {max_lines} lines) ===\n{'='*60}\n"
                return header + content
            else:
                error_msg = result.stderr or "Unknown error"
                return f"Error reading {file_path}: {error_msg}"
        
        except subprocess.TimeoutExpired:
            return f"Error: Timeout reading {file_path}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def process_result(
        self,
        command: str,
        result_text: str,
        task_instruction: str = "",
        max_files: int = 3,
        max_lines_per_file: int = 100
    ) -> tuple[str, List[str]]:
        """
        处理命令结果，智能选择并读取CSV文件
        
        Args:
            command: 原始命令
            result_text: 命令执行结果
            task_instruction: 任务指引（用于LLM判断）
            max_files: 最多读取文件数
            max_lines_per_file: 每个文件最大读取行数
            
        Returns:
            (增强后的结果, 读取的文件路径列表)
        """
        # 检测CSV路径 (使用LLM)
        file_infos = await self.detect_csv_paths(result_text, task_instruction)
        
        if not file_infos:
            # 没有检测到CSV文件，直接返回原结果
            return result_text, []
        
        self.agent_logger.info(f"🔍 Found {len(file_infos)} CSV file(s)")
        
        # 使用LLM选择最相关的文件
        selected_paths = await self.select_files(file_infos, task_instruction, max_files)
        
        if not selected_paths:
            return result_text, []
        
        # 读取选中的文件内容
        file_contents = []
        read_files = []
        
        for file_path in selected_paths:
            self.agent_logger.info(f"📖 Reading: {os.path.basename(file_path)}")
            content = self.read_file(file_path, max_lines_per_file)
            file_contents.append(content)
            read_files.append(file_path)
        
        # 组合结果
        enhanced_result = result_text + "\n\n" + "="*60 + "\n"
        enhanced_result += f"📊 Auto-read {len(read_files)}/{len(file_infos)} selected file(s):\n"
        enhanced_result += "="*60
        enhanced_result += "\n".join(file_contents)
        
        if len(file_infos) > len(read_files):
            enhanced_result += f"\n\n... and {len(file_infos) - len(read_files)} more file(s) not selected"
        
        return enhanced_result, read_files
    
    def should_read_files(self, command: str) -> bool:
        """
        判断是否应该读取文件
        
        Args:
            command: 执行的命令
            
        Returns:
            是否应该读取文件
        """
        # 只对特定命令读取文件
        file_commands = [
            'get_traces',
            'get_metrics',
        ]
        
        return any(cmd in command for cmd in file_commands)

