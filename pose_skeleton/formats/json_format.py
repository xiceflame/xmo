"""
JSON格式转换器

处理JSON格式的动作捕捉数据
支持多种常见的JSON格式：
1. 标准格式（自定义）
2. MediaPipe格式
3. OpenPose格式
"""

import json
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
from .base_format import BaseFormat
from ..data_structures import SkeletonSequence, Skeleton, KeyPoint

class JSONFormat(BaseFormat):
    """JSON格式转换器"""
    
    def __init__(self):
        """初始化JSON格式转换器"""
        self.format_handlers = {
            'standard': self._load_standard_format,
            'mediapipe': self._load_mediapipe_format,
            'openpose': self._load_openpose_format
        }
    
    def load(self, source: str) -> SkeletonSequence:
        """
        从JSON文件加载动作数据
        
        Args:
            source: JSON文件路径
            
        Returns:
            SkeletonSequence: 转换后的骨骼序列
        """
        with open(source, 'r') as f:
            data = json.load(f)
            
        # 检测JSON格式类型
        format_type = self._detect_format(data)
        handler = self.format_handlers.get(format_type)
        
        if handler is None:
            raise ValueError(f"Unsupported JSON format: {format_type}")
            
        return handler(data)
    
    def save(self, skeleton_sequence: SkeletonSequence, target: str) -> None:
        """
        将骨骼序列保存为JSON格式
        
        Args:
            skeleton_sequence: 要保存的骨骼序列
            target: 目标JSON文件路径
        """
        # 转换为标准格式
        data = {
            'format': 'standard',
            'version': '1.0',
            'fps': skeleton_sequence.fps,
            'frames': []
        }
        
        for skeleton in skeleton_sequence.frames:
            frame_data = {
                'frame_id': skeleton.metadata.get('frame_id', 0),
                'keypoints': []
            }
            
            for landmark_id, keypoint in skeleton.keypoints.items():
                frame_data['keypoints'].append({
                    'id': landmark_id,
                    'name': keypoint.name,
                    'position': keypoint.position.tolist(),
                    'confidence': keypoint.confidence
                })
                
            data['frames'].append(frame_data)
            
        with open(target, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self, data: str) -> bool:
        """
        验证JSON文件格式
        
        Args:
            data: JSON文件路径
            
        Returns:
            bool: 文件格式是否有效
        """
        try:
            with open(data, 'r') as f:
                json_data = json.load(f)
            return self._detect_format(json_data) is not None
        except Exception:
            return False
    
    def _detect_format(self, data: Dict) -> Optional[str]:
        """
        检测JSON数据的格式类型
        
        Args:
            data: JSON数据
            
        Returns:
            str: 格式类型
        """
        if 'format' in data and data['format'] == 'standard':
            return 'standard'
        elif 'pose_landmarks' in data:
            return 'mediapipe'
        elif 'people' in data and 'pose_keypoints_2d' in data['people'][0]:
            return 'openpose'
        return None
    
    def _load_standard_format(self, data: Dict) -> SkeletonSequence:
        """加载标准格式"""
        sequence = SkeletonSequence()
        sequence.fps = data.get('fps', 30.0)
        
        for frame_data in data['frames']:
            skeleton = Skeleton('json')
            skeleton.metadata['frame_id'] = frame_data['frame_id']
            
            for kp_data in frame_data['keypoints']:
                keypoint = KeyPoint(
                    landmark_id=kp_data['id'],
                    name=kp_data['name'],
                    position=np.array(kp_data['position']),
                    confidence=kp_data['confidence']
                )
                skeleton.add_keypoint(keypoint)
                
            sequence.add_frame(skeleton)
            
        return sequence
    
    def _load_mediapipe_format(self, data: Dict) -> SkeletonSequence:
        """加载MediaPipe格式"""
        sequence = SkeletonSequence()
        sequence.fps = data.get('fps', 30.0)
        
        for frame_id, frame_data in enumerate(data.get('pose_landmarks', [])):
            skeleton = Skeleton('mediapipe')
            skeleton.metadata['frame_id'] = frame_id
            
            for landmark_id, landmark in enumerate(frame_data.get('landmark', [])):
                keypoint = KeyPoint(
                    landmark_id=landmark_id,
                    name=f'landmark_{landmark_id}',
                    position=np.array([landmark['x'], landmark['y'], landmark['z']]),
                    confidence=landmark.get('visibility', 1.0)
                )
                skeleton.add_keypoint(keypoint)
                
            sequence.add_frame(skeleton)
            
        return sequence
    
    def _load_openpose_format(self, data: Dict) -> SkeletonSequence:
        """加载OpenPose格式"""
        sequence = SkeletonSequence()
        sequence.fps = data.get('fps', 30.0)
        
        for frame_id, person_data in enumerate(data.get('people', [])):
            skeleton = Skeleton('openpose')
            skeleton.metadata['frame_id'] = frame_id
            
            keypoints_2d = person_data['pose_keypoints_2d']
            for i in range(0, len(keypoints_2d), 3):
                landmark_id = i // 3
                keypoint = KeyPoint(
                    landmark_id=landmark_id,
                    name=f'landmark_{landmark_id}',
                    position=np.array([
                        keypoints_2d[i],
                        keypoints_2d[i + 1],
                        0.0  # OpenPose 2D只有x,y坐标
                    ]),
                    confidence=keypoints_2d[i + 2]
                )
                skeleton.add_keypoint(keypoint)
                
            sequence.add_frame(skeleton)
            
        return sequence
