"""
Prompt loader for managing system and user prompts
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from string import Template


class PromptLoader:
    """统一的提示词加载器"""

    def __init__(self, prompts_dir: Optional[str] = None):
        """
        初始化提示词加载器

        Args:
            prompts_dir: 提示词目录路径
        """
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            # 默认使用当前文件所在目录
            self.prompts_dir = Path(__file__).parent

        self._cache = {}

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        加载YAML文件

        Args:
            filename: YAML文件名

        Returns:
            解析后的字典
        """
        # 检查缓存
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.prompts_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)

        # 缓存结果
        self._cache[filename] = content
        return content

    def get_prompt(self,
                   agent_type: str,
                   prompt_type: str = "system",
                   **kwargs) -> str:
        """
        获取特定智能体的提示词

        Args:
            agent_type: 智能体类型 (observer/probe/executor/compressor)
            prompt_type: 提示词类型 (system/user)
            **kwargs: 用于填充模板的参数

        Returns:
            格式化后的提示词
        """
        filename = f"{agent_type}_prompts.yaml"
        prompts = self.load_yaml(filename)

        # 获取对应类型的提示词
        prompt_template = prompts.get(f"{prompt_type}_prompt", "")

        # 使用Template进行安全的字符串替换
        if kwargs:
            template = Template(prompt_template)
            # 安全替换 - 忽略未提供的变量
            prompt = template.safe_substitute(**kwargs)
        else:
            prompt = prompt_template

        return prompt

    def get_examples(self, agent_type: str) -> Dict[str, str]:
        """
        获取智能体的示例

        Args:
            agent_type: 智能体类型

        Returns:
            示例字典
        """
        filename = f"{agent_type}_prompts.yaml"
        prompts = self.load_yaml(filename)
        return prompts.get("examples", {})

    def get_rules(self, agent_type: str) -> Dict[str, Any]:
        """
        获取智能体的规则

        Args:
            agent_type: 智能体类型

        Returns:
            规则字典
        """
        filename = f"{agent_type}_prompts.yaml"
        prompts = self.load_yaml(filename)
        return prompts.get("rules", {})


# 单例模式的全局加载器
_default_loader = None


def get_prompt_loader() -> PromptLoader:
    """获取默认的提示词加载器"""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader