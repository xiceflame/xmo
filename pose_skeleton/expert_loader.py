"""
专家数据加载模块

负责从视频文件或Parquet文件加载专家动作数据，主要功能：
1. 视频帧提取和处理
2. 姿势检测和关键点提取
3. 数据预处理和清洗
4. 数据格式转换和存储
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union, Generator
from datetime import datetime

from .source import VideoSource, ParquetSource
from .detector import MediaPipePoseDetector
from .loader import BatchPoseLoader
from .data_structures import SkeletonSequence
from .data_format import DataFormat

class ExpertLoader:
    """专家数据加载器，包装了BatchPoseLoader"""
    
    def __init__(self,
                 data_root: str = "/mnt/qh/projects/aiyoga-CHI/data/expert/poses",
                 buffer_size: int = 30,
                 smooth_window: int = 3):
        """
        初始化专家数据加载器
        
        Args:
            data_root: 数据根目录
            buffer_size: 缓冲区大小
            smooth_window: 平滑窗口大小
        """
        self.data_root = Path(data_root)
        self.buffer_size = buffer_size
        self.smooth_window = smooth_window
        self.data_format = DataFormat()
        
        # 创建检测器
        self.detector = MediaPipePoseDetector(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 创建批处理加载器
        self.loader = BatchPoseLoader(
            pose_source=None,  # 将在加载时设置
            pose_detector=self.detector,
            buffer_size=buffer_size,
            smooth_window=smooth_window
        )
        
    def load_from_video(self, video_path: Union[str, Path], pose_name: str) -> Optional[SkeletonSequence]:
        """
        从视频文件加载专家数据
        
        Args:
            video_path: 视频文件路径
            pose_name: 姿势名称
            
        Returns:
            SkeletonSequence对象，如果加载失败则返回None
        """
        sequence = self.loader.process_video(video_path)
        if sequence is not None:
            sequence.metadata.update({
                'source_type': 'expert_video',
                'pose_name': pose_name,
                'video_path': str(video_path)
            })
        return sequence
        
    def process_directory(self,
                         directory: Union[str, Path],
                         pattern: str = "*.mp4") -> Dict[str, SkeletonSequence]:
        """
        批量处理目录
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            
        Returns:
            文件名到SkeletonSequence的映射
        """
        sequences = self.loader.process_directory(directory, pattern)
        
        # 添加专家数据特有的元数据
        for name, sequence in sequences.items():
            sequence.metadata.update({
                'source_type': 'expert_video',
                'pose_name': Path(name).stem
            })
            
        return sequences
        
    def save_to_parquet(self, sequence: SkeletonSequence, output_path: Union[str, Path]):
        """
        将序列保存为parquet格式
        
        Args:
            sequence: SkeletonSequence对象
            output_path: 输出文件路径
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为DataFrame
            df = self.data_format.sequence_to_df(sequence)
            
            # 添加元数据列
            for key, value in sequence.metadata.items():
                if isinstance(value, (str, int, float)):
                    df[key] = value
                
            # 保存为parquet
            df.to_parquet(output_path, index=False)
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving to parquet {output_path}: {str(e)}")
            
    def load_from_parquet(self, file_path: Union[str, Path]) -> Optional[SkeletonSequence]:
        """
        从parquet文件加载数据
        
        Args:
            file_path: parquet文件路径
            
        Returns:
            SkeletonSequence对象，如果加载失败则返回None
        """
        try:
            parquet_source = ParquetSource(file_path)
            original_source = self.loader.pose_source
            self.loader.pose_source = parquet_source
            
            if not self.loader.initialize():
                self.loader.pose_source = original_source
                return None
                
            sequence = self.loader.get_sequence()
            if sequence is not None:
                # 从DataFrame中提取元数据
                df = pd.read_parquet(file_path)
                metadata_cols = [col for col in df.columns if col not in self.data_format.landmark_columns]
                for col in metadata_cols:
                    sequence.metadata[col] = df[col].iloc[0]
                    
            self.loader.pose_source = original_source
            return sequence
            
        except Exception as e:
            print(f"Error loading from parquet {file_path}: {str(e)}")
            return None
