"""
基础格式转换接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..data_structures import SkeletonSequence

class BaseFormat(ABC):
    """基础格式转换器接口"""
    
    @abstractmethod
    def load(self, source: str) -> SkeletonSequence:
        """
        从源文件加载数据并转换为SkeletonSequence
        
        Args:
            source: 源文件路径
            
        Returns:
            SkeletonSequence: 转换后的骨骼序列
        """
        pass
    
    @abstractmethod
    def save(self, skeleton_sequence: SkeletonSequence, target: str) -> None:
        """
        将SkeletonSequence保存为目标格式
        
        Args:
            skeleton_sequence: 要保存的骨骼序列
            target: 目标文件路径
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        验证数据格式是否有效
        
        Args:
            data: 要验证的数据
            
        Returns:
            bool: 数据格式是否有效
        """
        pass
