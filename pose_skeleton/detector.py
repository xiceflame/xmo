"""
姿势检测器模块

提供不同类型的姿势检测器实现，包括：
- MediaPipe
- OpenPose（待实现）
- 其他可能的检测器
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

class PoseDetector(ABC):
    """姿势检测器抽象基类"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化检测器"""
        pass
        
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """检测姿势"""
        pass
        
    @abstractmethod
    def release(self):
        """释放资源"""
        pass

class MediaPipePoseDetector(PoseDetector):
    """MediaPipe姿势检测器"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_pose = mp.solutions.pose
        self.pose = None
        
    def initialize(self) -> bool:
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            return True
        except Exception as e:
            print(f"Error initializing MediaPipe pose detector: {str(e)}")
            return False
            
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        if self.pose is None:
            return None
            
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks is None:
                return None
                
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
                
            return landmarks
            
        except Exception as e:
            print(f"Error detecting pose: {str(e)}")
            return None
            
    def release(self):
        if self.pose is not None:
            self.pose.close()

# TODO: Add OpenPose detector implementation
class OpenPosePoseDetector(PoseDetector):
    """OpenPose姿势检测器（待实现）"""
    pass
