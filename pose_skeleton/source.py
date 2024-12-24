"""
姿势数据源模块

提供不同类型的数据源实现，包括：
- 视频文件
- 实时摄像头
- Parquet文件
- 其他可能的数据源
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from abc import ABC, abstractmethod

class PoseSource(ABC):
    """姿势数据源抽象基类"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化数据源"""
        pass
        
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取下一帧"""
        pass
        
    @abstractmethod
    def release(self):
        """释放资源"""
        pass
        
    @property
    @abstractmethod
    def fps(self) -> float:
        """获取帧率"""
        pass
        
    @property
    @abstractmethod
    def frame_count(self) -> int:
        """获取总帧数"""
        pass

class VideoSource(PoseSource):
    """视频文件数据源"""
    
    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self.cap = None
        self._fps = 0
        self._frame_count = 0
        
    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return self.cap.isOpened()
        except Exception as e:
            print(f"Error initializing video source: {str(e)}")
            return False
            
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            
    @property
    def fps(self) -> float:
        return self._fps
        
    @property
    def frame_count(self) -> int:
        return self._frame_count

class CameraSource(PoseSource):
    """实时摄像头数据源"""
    
    def __init__(self, camera_id: int = 0, target_fps: float = 30.0):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.cap = None
        self._frame_count = 0
        
    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            return self.cap.isOpened()
        except Exception as e:
            print(f"Error initializing camera source: {str(e)}")
            return False
            
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            
    @property
    def fps(self) -> float:
        return self.target_fps
        
    @property
    def frame_count(self) -> int:
        return self._frame_count

class ParquetSource(PoseSource):
    """Parquet文件数据源"""
    
    def __init__(self, file_path: Union[str, Path], fps: float = 30.0):
        self.file_path = Path(file_path)
        self._fps = fps
        self.data = None
        self.current_frame = 0
        
    def initialize(self) -> bool:
        try:
            self.data = pd.read_parquet(self.file_path)
            self._frame_count = self.data['frame'].nunique()
            return True
        except Exception as e:
            print(f"Error initializing parquet source: {str(e)}")
            return False
            
    def read_frame(self) -> Tuple[bool, Optional[Dict]]:
        if self.data is None or self.current_frame >= self._frame_count:
            return False, None
            
        frame_data = self.data[self.data['frame'] == self.current_frame]
        self.current_frame += 1
        return True, frame_data
        
    def release(self):
        self.data = None
        self.current_frame = 0
        
    @property
    def fps(self) -> float:
        return self._fps
        
    @property
    def frame_count(self) -> int:
        return self._frame_count if self.data is not None else 0
