"""
学生动作加载器模块

该模块提供了加载和处理学生动作数据的功能，支持：
1. 实时摄像头输入
2. 视频文件输入
3. 批量处理
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Generator
from datetime import datetime

from .source import VideoSource, CameraSource
from .detector import MediaPipePoseDetector
from .loader import RealtimePoseLoader, BatchPoseLoader
from .data_structures import SkeletonSequence

class StudentLoader:
    """学生动作加载器，包装了RealtimePoseLoader和BatchPoseLoader"""
    
    def __init__(self,
                 use_camera: bool = True,
                 camera_id: int = 0,
                 target_fps: float = 30.0,
                 buffer_size: int = 30,
                 smooth_window: int = 3):
        """
        初始化学生动作加载器
        
        Args:
            use_camera: 是否使用摄像头
            camera_id: 摄像头ID
            target_fps: 目标帧率
            buffer_size: 缓冲区大小
            smooth_window: 平滑窗口大小
        """
        self.use_camera = use_camera
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.smooth_window = smooth_window
        
        # 创建检测器
        self.detector = MediaPipePoseDetector()
        
        # 根据输入源创建加载器
        if use_camera:
            source = CameraSource(camera_id, target_fps)
            self.loader = RealtimePoseLoader(
                pose_source=source,
                pose_detector=self.detector,
                buffer_size=buffer_size,
                smooth_window=smooth_window
            )
        else:
            self.loader = BatchPoseLoader(
                pose_source=None,  # 将在加载视频时设置
                pose_detector=self.detector,
                buffer_size=buffer_size,
                smooth_window=smooth_window
            )
            
    def initialize(self) -> bool:
        """初始化加载器"""
        return self.loader.initialize()
        
    def release(self):
        """释放资源"""
        self.loader.release()
        
    def stream(self) -> Generator[Optional[SkeletonSequence], None, None]:
        """流式处理（用于实时输入）"""
        if not isinstance(self.loader, RealtimePoseLoader):
            raise RuntimeError("Stream mode is only available for camera input")
            
        for sequence in self.loader.stream():
            if sequence is not None:
                sequence.metadata.update({
                    'source_type': 'student_camera',
                    'camera_id': self.camera_id,
                    'target_fps': self.target_fps
                })
            yield sequence
            
    def process_video(self, video_path: Union[str, Path]) -> Optional[SkeletonSequence]:
        """处理视频文件"""
        if isinstance(self.loader, RealtimePoseLoader):
            raise RuntimeError("Video processing is not available in camera mode")
            
        sequence = self.loader.process_video(video_path)
        if sequence is not None:
            sequence.metadata.update({
                'source_type': 'student_video',
                'target_fps': self.target_fps
            })
        return sequence
        
    def process_directory(self,
                         directory: Union[str, Path],
                         pattern: str = "*.mp4") -> Dict[str, SkeletonSequence]:
        """批量处理目录"""
        if isinstance(self.loader, RealtimePoseLoader):
            raise RuntimeError("Directory processing is not available in camera mode")
            
        return self.loader.process_directory(directory, pattern)
