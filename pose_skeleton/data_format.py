"""
数据格式转换模块

提供在不同数据格式之间转换的工具：
1. 基础格式（DataFrame、Skeleton）
2. 动作捕捉格式（BVH、FBX）
3. 通用格式（JSON、CSV）
4. 特定格式（MediaPipe）

主要功能：
- 统一的格式转换入口
- 多格式支持和转换
- 数据验证和错误处理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Type
from pathlib import Path
from .data_structures import Skeleton, SkeletonSequence, KeyPoint, Connection
from .formats import (
    BaseFormat,
    BVHFormat,
    FBXFormat,
    JSONFormat,
    CSVFormat,
    MediaPipeFormat
)

class DataFormatManager:
    """数据格式转换管理器"""
    
    def __init__(self):
        """初始化格式转换管理器"""
        # 基础映射定义（从原有代码移植）
        self.landmarks = {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        self.landmark_names = {v: k for k, v in self.landmarks.items()}
        
        self.bones = {
            'torso': [
                ('left_shoulder', 'right_shoulder'),
                ('left_hip', 'right_hip'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip')
            ],
            'left_arm': [
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('left_wrist', 'left_thumb'),
                ('left_wrist', 'left_pinky'),
                ('left_wrist', 'left_index')
            ],
            'right_arm': [
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('right_wrist', 'right_thumb'),
                ('right_wrist', 'right_pinky'),
                ('right_wrist', 'right_index')
            ],
            'left_leg': [
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('left_ankle', 'left_heel'),
                ('left_ankle', 'left_foot_index')
            ],
            'right_leg': [
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle'),
                ('right_ankle', 'right_heel'),
                ('right_ankle', 'right_foot_index')
            ]
        }
        
        # 注册格式转换器
        self._format_handlers = {
            'bvh': BVHFormat(),
            'fbx': FBXFormat(),
            'json': JSONFormat(),
            'csv': CSVFormat(),
            'mediapipe': MediaPipeFormat()
        }
    
    def load_file(self, file_path: str, format_type: Optional[str] = None) -> SkeletonSequence:
        """
        从文件加载动作数据
        
        Args:
            file_path: 源文件路径
            format_type: 文件格式类型，如果为None则自动检测
            
        Returns:
            SkeletonSequence: 加载的骨骼序列
        """
        if format_type is None:
            format_type = Path(file_path).suffix[1:].lower()
            
        handler = self._format_handlers.get(format_type)
        if handler is None:
            raise ValueError(f"Unsupported format type: {format_type}")
            
        return handler.load(file_path)
    
    def save_file(self, sequence: SkeletonSequence, file_path: str, format_type: Optional[str] = None) -> None:
        """
        保存动作数据到文件
        
        Args:
            sequence: 要保存的骨骼序列
            file_path: 目标文件路径
            format_type: 文件格式类型，如果为None则从文件扩展名推断
        """
        if format_type is None:
            format_type = Path(file_path).suffix[1:].lower()
            
        handler = self._format_handlers.get(format_type)
        if handler is None:
            raise ValueError(f"Unsupported format type: {format_type}")
            
        handler.save(sequence, file_path)
    
    def convert_format(self, source_file: str, target_file: str,
                      source_format: Optional[str] = None,
                      target_format: Optional[str] = None) -> None:
        """
        在不同格式之间转换
        
        Args:
            source_file: 源文件路径
            target_file: 目标文件路径
            source_format: 源文件格式，如果为None则自动检测
            target_format: 目标文件格式，如果为None则自动检测
        """
        sequence = self.load_file(source_file, source_format)
        self.save_file(sequence, target_file, target_format)
    
    def long_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将长格式DataFrame转换为宽格式
        
        Args:
            df: 长格式DataFrame，包含frame, landmark_index, x, y, z列
            
        Returns:
            宽格式DataFrame，每行是一帧，列是所有关键点的x,y,z坐标
        """
        # 确保必要的列存在
        required_cols = ['frame', 'landmark_index', 'x', 'y', 'z']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
        # 重塑数据
        coords = []
        for coord in ['x', 'y', 'z']:
            # 透视转换
            coord_df = df.pivot(index='frame', columns='landmark_index', values=coord)
            # 重命名列
            coord_df.columns = [f'{coord}{i}' for i in range(len(coord_df.columns))]
            coords.append(coord_df)
            
        # 合并所有坐标
        return pd.concat(coords, axis=1)
        
    def wide_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将宽格式DataFrame转换为长格式
        
        Args:
            df: 宽格式DataFrame，列名格式为x0,y0,z0,x1,y1,z1,...
            
        Returns:
            长格式DataFrame，包含frame, landmark_index, x, y, z列
        """
        n_landmarks = len(df.columns) // 3
        frames = []
        
        for frame_id, row in df.iterrows():
            frame_data = []
            for i in range(n_landmarks):
                frame_data.append({
                    'frame': frame_id,
                    'landmark_index': i,
                    'x': row[f'x{i}'],
                    'y': row[f'y{i}'],
                    'z': row[f'z{i}']
                })
            frames.extend(frame_data)
            
        return pd.DataFrame(frames)
        
    def df_to_skeleton(self, df: pd.DataFrame, frame_id: int, source_type: str = 'expert') -> Skeleton:
        """
        将DataFrame的一帧数据转换为Skeleton对象
        
        Args:
            df: 长格式DataFrame
            frame_id: 帧ID
            source_type: 数据来源类型
            
        Returns:
            Skeleton对象
        """
        # 创建骨骼对象
        skeleton = Skeleton(source_type)
        skeleton.metadata['frame_id'] = frame_id
        
        # 获取当前帧的数据
        frame_data = df[df['frame'] == frame_id]
        
        # 添加关键点
        for _, row in frame_data.iterrows():
            landmark_id = row['landmark_index']
            landmark_name = self.landmark_names.get(landmark_id, f'landmark_{landmark_id}')
            
            keypoint = KeyPoint(
                landmark_id=landmark_id,
                name=landmark_name,
                position=np.array([row['x'], row['y'], row['z']]),
                confidence=1.0 if source_type == 'expert' else row.get('confidence', 0.8)
            )
            skeleton.add_keypoint(keypoint)
            
        # 添加骨骼连接
        for part_name, connections in self.bones.items():
            for start_name, end_name in connections:
                start_id = self.landmarks[start_name]
                end_id = self.landmarks[end_name]
                
                if start_id in skeleton.keypoints and end_id in skeleton.keypoints:
                    connection = Connection(
                        start_point=start_id,
                        end_point=end_id,
                        connection_type=part_name
                    )
                    skeleton.add_connection(connection)
                    
        return skeleton
        
    def df_to_sequence(self, df: pd.DataFrame, source_type: str = 'expert', fps: float = 30.0) -> SkeletonSequence:
        """
        将整个DataFrame转换为SkeletonSequence对象
        
        Args:
            df: 长格式DataFrame
            source_type: 数据来源类型
            fps: 帧率
            
        Returns:
            SkeletonSequence对象
        """
        sequence = SkeletonSequence()
        sequence.fps = fps
        
        # 按帧转换
        for frame_id in sorted(df['frame'].unique()):
            skeleton = self.df_to_skeleton(df, frame_id, source_type)
            sequence.add_frame(skeleton)
            
        return sequence
        
    def skeleton_to_df(self, skeleton: Skeleton) -> pd.DataFrame:
        """
        将Skeleton对象转换为DataFrame
        
        Args:
            skeleton: Skeleton对象
            
        Returns:
            长格式DataFrame
        """
        frame_id = skeleton.metadata.get('frame_id', 0)
        rows = []
        
        for landmark_id, keypoint in skeleton.keypoints.items():
            rows.append({
                'frame': frame_id,
                'landmark_index': landmark_id,
                'x': keypoint.position[0],
                'y': keypoint.position[1],
                'z': keypoint.position[2],
                'confidence': keypoint.confidence
            })
            
        return pd.DataFrame(rows)
        
    def sequence_to_df(self, sequence: SkeletonSequence) -> pd.DataFrame:
        """
        将SkeletonSequence对象转换为DataFrame
        
        Args:
            sequence: SkeletonSequence对象
            
        Returns:
            长格式DataFrame
        """
        dfs = []
        for skeleton in sequence.frames:
            df = self.skeleton_to_df(skeleton)
            dfs.append(df)
            
        return pd.concat(dfs, ignore_index=True)
