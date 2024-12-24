"""
姿势加载器模块

提供不同类型的姿势加载器实现，包括：
- 基础加载器
- 实时加载器
- 批量加载器
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Generator
from datetime import datetime
from abc import ABC, abstractmethod

from .source import PoseSource, VideoSource, CameraSource, ParquetSource
from .detector import PoseDetector, MediaPipePoseDetector
from .data_structures import KeyPoint, Connection, Skeleton, SkeletonSequence
from .data_format import DataFormat

class BasePoseLoader:
    """基础姿势加载器"""
    
    def __init__(self,
                 pose_source: PoseSource,
                 pose_detector: Optional[PoseDetector] = None,
                 buffer_size: int = 30,
                 smooth_window: int = 3):
        """
        初始化基础加载器
        
        Args:
            pose_source: 姿势数据源
            pose_detector: 姿势检测器（对于视频源必需）
            buffer_size: 缓冲区大小
            smooth_window: 平滑窗口大小
        """
        self.pose_source = pose_source
        self.pose_detector = pose_detector
        self.buffer_size = buffer_size
        self.smooth_window = smooth_window
        self.data_format = DataFormat()
        self.frame_buffer = []
        self.landmarks_buffer = []
        
    def initialize(self) -> bool:
        """初始化加载器"""
        source_ok = self.pose_source.initialize()
        if self.pose_detector is not None:
            return source_ok and self.pose_detector.initialize()
        return source_ok
        
    def release(self):
        """释放资源"""
        self.pose_source.release()
        if self.pose_detector is not None:
            self.pose_detector.release()
            
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """处理单帧"""
        if self.pose_detector is None:
            return None
            
        # 检测姿势
        landmarks = self.pose_detector.detect(frame)
        if landmarks is None:
            return None
            
        # 更新缓冲区
        self.frame_buffer.append(frame)
        self.landmarks_buffer.append(landmarks)
        
        # 保持缓冲区大小
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
            self.landmarks_buffer.pop(0)
            
        # 平滑处理
        if len(self.landmarks_buffer) >= self.smooth_window:
            landmarks = self._smooth_landmarks()
            
        return landmarks
        
    def _smooth_landmarks(self) -> Dict:
        """平滑关键点数据"""
        smoothed = {}
        window = self.landmarks_buffer[-self.smooth_window:]
        
        # 对每个关键点进行平滑
        for idx in window[0].keys():
            x = np.mean([frame[idx]['x'] for frame in window])
            y = np.mean([frame[idx]['y'] for frame in window])
            z = np.mean([frame[idx]['z'] for frame in window])
            visibility = np.mean([frame[idx]['visibility'] for frame in window])
            
            smoothed[idx] = {
                'x': x,
                'y': y,
                'z': z,
                'visibility': visibility
            }
            
        return smoothed
        
    def get_skeleton(self, landmarks: Dict) -> Optional[Skeleton]:
        """转换为骨骼数据结构"""
        try:
            keypoints = []
            for idx, data in landmarks.items():
                keypoint = KeyPoint(
                    index=idx,
                    x=data['x'],
                    y=data['y'],
                    z=data['z'],
                    visibility=data['visibility']
                )
                keypoints.append(keypoint)
                
            return Skeleton(keypoints=keypoints)
            
        except Exception as e:
            print(f"Error converting to skeleton: {str(e)}")
            return None
            
    def get_sequence(self) -> Optional[SkeletonSequence]:
        """获取当前缓冲区的骨骼序列"""
        if not self.landmarks_buffer:
            return None
            
        try:
            skeletons = []
            for landmarks in self.landmarks_buffer:
                skeleton = self.get_skeleton(landmarks)
                if skeleton is not None:
                    skeletons.append(skeleton)
                    
            if not skeletons:
                return None
                
            sequence = SkeletonSequence(
                skeletons=skeletons,
                fps=self.pose_source.fps
            )
            
            return sequence
            
        except Exception as e:
            print(f"Error creating sequence: {str(e)}")
            return None

class RealtimePoseLoader(BasePoseLoader):
    """实时姿势加载器"""
    
    def stream(self) -> Generator[Optional[SkeletonSequence], None, None]:
        """流式处理"""
        if not self.initialize():
            return
            
        try:
            while True:
                ret, frame = self.pose_source.read_frame()
                if not ret:
                    break
                    
                landmarks = self.process_frame(frame)
                if landmarks is not None:
                    sequence = self.get_sequence()
                    if sequence is not None:
                        sequence.metadata.update({
                            'source_type': 'realtime',
                            'buffer_size': self.buffer_size,
                            'smooth_window': self.smooth_window,
                            'processed_time': datetime.now().isoformat()
                        })
                    yield sequence
                    
        finally:
            self.release()

class BatchPoseLoader(BasePoseLoader):
    """批量姿势加载器"""
    
    def process_video(self, video_path: Union[str, Path]) -> Optional[SkeletonSequence]:
        """处理视频文件"""
        video_source = VideoSource(video_path)
        original_source = self.pose_source
        self.pose_source = video_source
        
        if not self.initialize():
            self.pose_source = original_source
            return None
            
        try:
            while True:
                ret, frame = self.pose_source.read_frame()
                if not ret:
                    break
                    
                self.process_frame(frame)
                
            sequence = self.get_sequence()
            if sequence is not None:
                sequence.metadata.update({
                    'source_type': 'video',
                    'source_path': str(video_path),
                    'buffer_size': self.buffer_size,
                    'smooth_window': self.smooth_window,
                    'processed_time': datetime.now().isoformat()
                })
            return sequence
            
        finally:
            self.release()
            self.pose_source = original_source
            
    def process_directory(self, 
                         directory: Union[str, Path],
                         pattern: str = "*.mp4") -> Dict[str, SkeletonSequence]:
        """批量处理目录"""
        directory = Path(directory)
        results = {}
        
        for video_file in directory.glob(pattern):
            sequence = self.process_video(video_file)
            if sequence is not None:
                results[video_file.name] = sequence
                
        return results
        
    def process_parquet(self, file_path: Union[str, Path]) -> Optional[SkeletonSequence]:
        """处理Parquet文件"""
        parquet_source = ParquetSource(file_path)
        original_source = self.pose_source
        self.pose_source = parquet_source
        
        if not self.initialize():
            self.pose_source = original_source
            return None
            
        try:
            while True:
                ret, frame_data = self.pose_source.read_frame()
                if not ret:
                    break
                    
                # TODO: Convert frame_data to landmarks format
                # self.process_frame(frame_data)
                
            sequence = self.get_sequence()
            if sequence is not None:
                sequence.metadata.update({
                    'source_type': 'parquet',
                    'source_path': str(file_path),
                    'processed_time': datetime.now().isoformat()
                })
            return sequence
            
        finally:
            self.release()
            self.pose_source = original_source
