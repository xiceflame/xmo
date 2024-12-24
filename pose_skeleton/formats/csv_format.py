"""
CSV格式转换器

处理CSV格式的动作捕捉数据，支持两种格式：
1. 宽格式：每行一帧，列为关键点坐标
2. 长格式：每行一个关键点，包含帧ID和坐标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from .base_format import BaseFormat
from ..data_structures import SkeletonSequence, Skeleton, KeyPoint

class CSVFormat(BaseFormat):
    """CSV格式转换器"""
    
    def __init__(self):
        """初始化CSV格式转换器"""
        self.required_wide_cols = ['frame']  # x0,y0,z0,x1,y1,z1,...
        self.required_long_cols = ['frame', 'landmark_index', 'x', 'y', 'z']
    
    def load(self, source: str) -> SkeletonSequence:
        """
        从CSV文件加载动作数据
        
        Args:
            source: CSV文件路径
            
        Returns:
            SkeletonSequence: 转换后的骨骼序列
        """
        df = pd.read_csv(source)
        
        # 检测CSV格式类型
        if self._is_wide_format(df):
            return self._load_wide_format(df)
        elif self._is_long_format(df):
            return self._load_long_format(df)
        else:
            raise ValueError("Unsupported CSV format")
    
    def save(self, skeleton_sequence: SkeletonSequence, target: str) -> None:
        """
        将骨骼序列保存为CSV格式
        
        Args:
            skeleton_sequence: 要保存的骨骼序列
            target: 目标CSV文件路径
        """
        # 转换为长格式DataFrame
        rows = []
        for frame_id, skeleton in enumerate(skeleton_sequence.frames):
            for landmark_id, keypoint in skeleton.keypoints.items():
                rows.append({
                    'frame': frame_id,
                    'landmark_index': landmark_id,
                    'name': keypoint.name,
                    'x': keypoint.position[0],
                    'y': keypoint.position[1],
                    'z': keypoint.position[2],
                    'confidence': keypoint.confidence
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(target, index=False)
    
    def validate(self, data: str) -> bool:
        """
        验证CSV文件格式
        
        Args:
            data: CSV文件路径
            
        Returns:
            bool: 文件格式是否有效
        """
        try:
            df = pd.read_csv(data)
            return self._is_wide_format(df) or self._is_long_format(df)
        except Exception:
            return False
    
    def _is_wide_format(self, df: pd.DataFrame) -> bool:
        """检查是否为宽格式"""
        if 'frame' not in df.columns:
            return False
            
        # 检查是否有x0,y0,z0等列
        coord_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]
        return len(coord_cols) > 0 and len(coord_cols) % 3 == 0
    
    def _is_long_format(self, df: pd.DataFrame) -> bool:
        """检查是否为长格式"""
        return all(col in df.columns for col in self.required_long_cols)
    
    def _load_wide_format(self, df: pd.DataFrame) -> SkeletonSequence:
        """加载宽格式数据"""
        sequence = SkeletonSequence()
        
        for _, row in df.iterrows():
            skeleton = Skeleton('csv')
            skeleton.metadata['frame_id'] = row['frame']
            
            # 获取所有坐标列
            coord_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]
            n_landmarks = len(coord_cols) // 3
            
            for i in range(n_landmarks):
                keypoint = KeyPoint(
                    landmark_id=i,
                    name=f'landmark_{i}',
                    position=np.array([
                        row[f'x{i}'],
                        row[f'y{i}'],
                        row[f'z{i}']
                    ]),
                    confidence=row.get(f'confidence{i}', 1.0)
                )
                skeleton.add_keypoint(keypoint)
                
            sequence.add_frame(skeleton)
            
        return sequence
    
    def _load_long_format(self, df: pd.DataFrame) -> SkeletonSequence:
        """加载长格式数据"""
        sequence = SkeletonSequence()
        
        for frame_id in sorted(df['frame'].unique()):
            skeleton = Skeleton('csv')
            skeleton.metadata['frame_id'] = frame_id
            
            frame_data = df[df['frame'] == frame_id]
            for _, row in frame_data.iterrows():
                keypoint = KeyPoint(
                    landmark_id=row['landmark_index'],
                    name=row.get('name', f'landmark_{row["landmark_index"]}'),
                    position=np.array([row['x'], row['y'], row['z']]),
                    confidence=row.get('confidence', 1.0)
                )
                skeleton.add_keypoint(keypoint)
                
            sequence.add_frame(skeleton)
            
        return sequence
