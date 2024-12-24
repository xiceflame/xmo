"""
MediaPipe格式转换器

处理MediaPipe姿势检测输出的数据格式
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base_format import BaseFormat
from ..data_structures import SkeletonSequence, Skeleton, KeyPoint

class MediaPipeFormat(BaseFormat):
    """MediaPipe格式转换器"""
    
    def __init__(self):
        """初始化MediaPipe格式转换器"""
        # MediaPipe姿势关键点映射
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
    
    def load(self, source: Any) -> SkeletonSequence:
        """
        从MediaPipe结果加载动作数据
        
        Args:
            source: MediaPipe姿势检测结果
            
        Returns:
            SkeletonSequence: 转换后的骨骼序列
        """
        sequence = SkeletonSequence()
        
        if isinstance(source, dict):  # 单帧结果
            skeleton = self._process_frame(source)
            sequence.add_frame(skeleton)
        elif isinstance(source, list):  # 多帧结果
            for frame_id, frame_data in enumerate(source):
                skeleton = self._process_frame(frame_data, frame_id)
                sequence.add_frame(skeleton)
                
        return sequence
    
    def save(self, skeleton_sequence: SkeletonSequence, target: str) -> None:
        """
        将骨骼序列保存为MediaPipe格式
        
        Args:
            skeleton_sequence: 要保存的骨骼序列
            target: 目标文件路径（不使用）
        """
        raise NotImplementedError("MediaPipe format does not support direct saving")
    
    def validate(self, data: Any) -> bool:
        """
        验证MediaPipe数据格式
        
        Args:
            data: MediaPipe数据
            
        Returns:
            bool: 数据格式是否有效
        """
        try:
            if isinstance(data, dict):
                return 'landmark' in data
            elif isinstance(data, list):
                return all('landmark' in frame for frame in data)
            return False
        except Exception:
            return False
    
    def _process_frame(self, frame_data: Dict, frame_id: int = 0) -> Skeleton:
        """
        处理单帧数据
        
        Args:
            frame_data: 帧数据
            frame_id: 帧ID
            
        Returns:
            Skeleton: 处理后的骨骼数据
        """
        skeleton = Skeleton('mediapipe')
        skeleton.metadata['frame_id'] = frame_id
        
        landmarks = frame_data.get('landmark', [])
        for i, landmark in enumerate(landmarks):
            keypoint = KeyPoint(
                landmark_id=i,
                name=self.landmark_names.get(i, f'landmark_{i}'),
                position=np.array([
                    landmark.get('x', 0.0),
                    landmark.get('y', 0.0),
                    landmark.get('z', 0.0)
                ]),
                confidence=landmark.get('visibility', 1.0)
            )
            skeleton.add_keypoint(keypoint)
            
        return skeleton
