"""
骨骼数据结构模块

提供统一的骨骼数据结构，支持：
1. 单帧骨骼数据（Skeleton）
2. 骨骼序列数据（SkeletonSequence）
3. 关键点和连接的定义
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import json

class KeyPoint:
    """关键点类"""
    def __init__(self, 
                 landmark_id: int,
                 name: str,
                 position: np.ndarray,
                 confidence: float = 1.0):
        """
        初始化关键点
        
        Args:
            landmark_id: 关键点ID
            name: 关键点名称
            position: 3D位置 [x, y, z]
            confidence: 置信度 (0-1)
            
        Raises:
            ValueError: 当参数无效时
        """
        # 验证参数
        if landmark_id < 0:
            raise ValueError("landmark_id必须大于等于0")
        if not isinstance(position, np.ndarray) or position.shape != (3,):
            raise ValueError("position必须是形状为(3,)的numpy数组")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence必须在0到1之间")
            
        self.id = landmark_id
        self.name = name
        self.position = position
        self.confidence = confidence
        
    def distance_to(self, other: 'KeyPoint') -> float:
        """计算到另一个关键点的欧氏距离"""
        return float(np.linalg.norm(self.position - other.position))
        
    def midpoint_with(self, other: 'KeyPoint') -> np.ndarray:
        """计算与另一个关键点的中点"""
        return (self.position + other.position) / 2
        
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'id': int(self.id),
            'name': self.name,
            'position': self.position.tolist(),
            'confidence': float(self.confidence)
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'KeyPoint':
        """从字典创建关键点"""
        return cls(
            landmark_id=data['id'],
            name=data['name'],
            position=np.array(data['position']),
            confidence=data['confidence']
        )

class Connection:
    """骨骼连接类"""
    def __init__(self,
                 start_point: Union[KeyPoint, int],
                 end_point: Union[KeyPoint, int],
                 connection_type: str = 'bone'):
        """
        初始化骨骼连接
        
        Args:
            start_point: 起始关键点或关键点ID
            end_point: 终止关键点或关键点ID
            connection_type: 连接类型（如'bone', 'auxiliary'等）
            
        Raises:
            ValueError: 当参数无效时
        """
        # 获取关键点ID
        start_id = start_point.id if isinstance(start_point, KeyPoint) else start_point
        end_id = end_point.id if isinstance(end_point, KeyPoint) else end_point
        
        # 验证参数
        if start_id < 0:
            raise ValueError("起始点ID必须大于等于0")
        if end_id < 0:
            raise ValueError("终止点ID必须大于等于0")
        if start_id == end_id:
            raise ValueError("起始点和终止点不能相同")
            
        self.start = start_id
        self.end = end_id
        self.type = connection_type
        
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'start': int(self.start),
            'end': int(self.end),
            'type': self.type
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Connection':
        """从字典创建连接"""
        return cls(
            start_point=data['start'],
            end_point=data['end'],
            connection_type=data['type']
        )

class Skeleton:
    """骨骼类，表示单帧的骨骼数据"""
    def __init__(self, source_type: str = 'unknown'):
        """
        初始化骨骼数据
        
        Args:
            source_type: 数据来源类型（'expert' 或 'student'）
        """
        self.keypoints: Dict[int, KeyPoint] = {}  # 关键点字典，key为landmark_id
        self.connections: List[Connection] = []  # 连接列表
        self.metadata = {
            'source_type': source_type,
            'timestamp': None,
            'frame_id': None,
            'confidence': None,
        }
        
    def add_keypoint(self, keypoint: KeyPoint):
        """
        添加关键点
        
        Raises:
            ValueError: 当关键点ID已存在时
        """
        if keypoint.id in self.keypoints:
            raise ValueError(f"关键点ID {keypoint.id} 已存在")
        self.keypoints[keypoint.id] = keypoint
        
    def add_connection(self, connection: Connection):
        """
        添加连接
        
        Raises:
            ValueError: 当连接的关键点不存在时
        """
        if connection.start not in self.keypoints or connection.end not in self.keypoints:
            raise ValueError("连接的关键点不存在")
        self.connections.append(connection)
        
    def get_keypoint_position(self, landmark_id: int) -> Optional[np.ndarray]:
        """获取关键点位置"""
        keypoint = self.keypoints.get(landmark_id)
        return keypoint.position if keypoint else None
        
    def get_connection_points(self, start_id: int, end_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取连接的两端点位置"""
        for connection in self.connections:
            if connection.start == start_id and connection.end == end_id:
                start_pos = self.get_keypoint_position(start_id)
                end_pos = self.get_keypoint_position(end_id)
                if start_pos is None or end_pos is None:
                    return None
                return start_pos, end_pos
        return None
        
    def calculate_bone_length(self, start_name: str, end_name: str) -> float:
        """
        计算骨骼长度
        
        Args:
            start_name: 起始关键点名称
            end_name: 终止关键点名称
            
        Returns:
            骨骼长度
            
        Raises:
            ValueError: 当关键点不存在时
        """
        # 查找关键点
        start_kp = None
        end_kp = None
        for kp in self.keypoints.values():
            if kp.name == start_name:
                start_kp = kp
            elif kp.name == end_name:
                end_kp = kp
                
        if start_kp is None:
            raise ValueError(f"找不到关键点: {start_name}")
        if end_kp is None:
            raise ValueError(f"找不到关键点: {end_name}")
            
        return start_kp.distance_to(end_kp)
        
    def calculate_joint_angle(self, point1_name: str, point2_name: str, point3_name: str) -> float:
        """
        计算关节角度（弧度）
        
        Args:
            point1_name: 第一个关键点名称
            point2_name: 中心关键点名称（关节点）
            point3_name: 第三个关键点名称
            
        Returns:
            角度（弧度）
            
        Raises:
            ValueError: 当关键点不存在时
        """
        # 查找关键点
        point1 = None
        point2 = None
        point3 = None
        for kp in self.keypoints.values():
            if kp.name == point1_name:
                point1 = kp
            elif kp.name == point2_name:
                point2 = kp
            elif kp.name == point3_name:
                point3 = kp
                
        if point1 is None:
            raise ValueError(f"找不到关键点: {point1_name}")
        if point2 is None:
            raise ValueError(f"找不到关键点: {point2_name}")
        if point3 is None:
            raise ValueError(f"找不到关键点: {point3_name}")
            
        # 计算向量
        vector1 = point1.position - point2.position
        vector2 = point3.position - point2.position
        
        # 计算角度
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 处理数值误差
        return np.arccos(cos_angle)
        
    def calculate_pose_center(self) -> np.ndarray:
        """
        计算姿势中心点（所有关键点的平均位置）
        
        Returns:
            中心点坐标
        """
        if not self.keypoints:
            raise ValueError("没有关键点")
            
        positions = np.array([kp.position for kp in self.keypoints.values()])
        return np.mean(positions, axis=0)
        
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'source_type': self.metadata['source_type'],
            'metadata': {k: int(v) if isinstance(v, np.integer) else v for k, v in self.metadata.items()},
            'keypoints': {int(k): v.to_dict() for k, v in self.keypoints.items()},
            'connections': [c.to_dict() for c in self.connections]
        }
        
    def to_normalized_format(self) -> Dict[str, np.ndarray]:
        """
        将骨骼数据转换为标准化格式
        
        Returns:
            包含关键点坐标的字典，格式为 {'landmark_name': np.array([x, y, z])}
        """
        return {name: kp.position for name, kp in self.keypoints.items()}
        
    def update_from_normalized(self, normalized_data: Dict[str, np.ndarray]):
        """
        从标准化数据更新骨骼
        
        Args:
            normalized_data: 包含标准化后关键点坐标的字典
        """
        for name, coords in normalized_data.items():
            if name in self.keypoints:
                self.keypoints[name].position = coords
                
    @classmethod
    def from_dict(cls, data: Dict) -> 'Skeleton':
        """从字典创建骨骼"""
        skeleton = cls(data['metadata']['source_type'])
        skeleton.metadata = data['metadata']
        
        # 加载关键点
        for _, kp_data in data['keypoints'].items():
            keypoint = KeyPoint.from_dict(kp_data)
            skeleton.add_keypoint(keypoint)
            
        # 加载连接
        for conn_data in data['connections']:
            connection = Connection.from_dict(conn_data)
            skeleton.add_connection(connection)
            
        return skeleton

class SkeletonSequence:
    """骨骼序列类，表示一系列连续的骨骼帧"""
    def __init__(self):
        """初始化骨骼序列"""
        self.frames: List[Skeleton] = []  # 骨骼帧列表
        self._fps: Optional[float] = None  # 帧率
        self.duration: Optional[float] = None  # 持续时间（秒）
        self.metadata: Dict = {}  # 序列元数据
        
    @property
    def fps(self) -> Optional[float]:
        """获取帧率"""
        return self._fps
        
    @fps.setter
    def fps(self, value: float):
        """
        设置帧率
        
        Raises:
            ValueError: 当帧率无效时
        """
        if value <= 0:
            raise ValueError("帧率必须大于0")
        self._fps = value
        if len(self.frames) > 1:
            self.duration = (len(self.frames) - 1) / value
        
    def add_frame(self, skeleton: Skeleton):
        """
        添加骨骼帧
        
        Raises:
            TypeError: 当输入不是Skeleton对象时
        """
        if not isinstance(skeleton, Skeleton):
            raise TypeError("输入必须是Skeleton对象")
            
        self.frames.append(skeleton)
        if self.fps and len(self.frames) > 1:
            self.duration = (len(self.frames) - 1) / self.fps
            
    def get_frame(self, index: int) -> Optional[Skeleton]:
        """获取指定索引的骨骼帧"""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None
        
    def get_frame_count(self) -> int:
        """获取总帧数"""
        return len(self.frames)
        
    def get_frame_slice(self, start: int, end: int) -> 'SkeletonSequence':
        """
        获取帧切片
        
        Args:
            start: 起始帧索引
            end: 结束帧索引
            
        Returns:
            新的SkeletonSequence对象
            
        Raises:
            ValueError: 当索引无效时
        """
        if not (0 <= start < len(self.frames)):
            raise ValueError("起始索引无效")
        if not (0 <= end <= len(self.frames)):
            raise ValueError("结束索引无效")
        if start >= end:
            raise ValueError("起始索引必须小于结束索引")
            
        new_seq = SkeletonSequence()
        new_seq.fps = self.fps
        new_seq.metadata = self.metadata.copy()
        new_seq.frames = self.frames[start:end]
        if new_seq.fps:
            new_seq.duration = (len(new_seq.frames) - 1) / new_seq.fps
        return new_seq
        
    def interpolate(self, target_fps: float) -> 'SkeletonSequence':
        """
        插值到目标帧率
        
        Args:
            target_fps: 目标帧率
            
        Returns:
            新的SkeletonSequence对象
            
        Raises:
            ValueError: 当目标帧率无效时
        """
        if target_fps <= 0:
            raise ValueError("目标帧率必须大于0")
        if not self.fps:
            raise ValueError("当前序列未设置帧率")
        if len(self.frames) < 2:
            raise ValueError("需要至少2帧才能进行插值")
            
        # 计算新的帧数
        old_duration = (len(self.frames) - 1) / self.fps
        new_frame_count = int(old_duration * target_fps) + 1
        
        # 创建新序列
        new_seq = SkeletonSequence()
        new_seq.fps = target_fps
        new_seq.metadata = self.metadata.copy()
        
        # 对每个时间点进行插值
        old_times = np.arange(len(self.frames)) / self.fps
        new_times = np.arange(new_frame_count) / target_fps
        
        for t in new_times:
            # 找到最近的两帧
            idx = np.searchsorted(old_times, t)
            if idx == 0:
                new_seq.add_frame(self.frames[0])
            elif idx >= len(self.frames):
                new_seq.add_frame(self.frames[-1])
            else:
                # 线性插值
                t1 = old_times[idx-1]
                t2 = old_times[idx]
                alpha = (t - t1) / (t2 - t1)
                
                frame1 = self.frames[idx-1]
                frame2 = self.frames[idx]
                
                # 创建插值帧
                interp_frame = Skeleton(frame1.metadata['source_type'])
                interp_frame.metadata = frame1.metadata.copy()
                
                # 插值关键点
                for kp_id, kp1 in frame1.keypoints.items():
                    kp2 = frame2.keypoints[kp_id]
                    pos = kp1.position * (1 - alpha) + kp2.position * alpha
                    conf = kp1.confidence * (1 - alpha) + kp2.confidence * alpha
                    interp_frame.add_keypoint(KeyPoint(
                        kp1.id,
                        kp1.name,
                        pos,
                        conf
                    ))
                
                # 复制连接
                for conn in frame1.connections:
                    interp_frame.add_connection(Connection(
                        conn.start,
                        conn.end,
                        conn.type
                    ))
                    
                new_seq.add_frame(interp_frame)
                
        return new_seq
        
    def to_normalized_format(self) -> List[Dict[str, np.ndarray]]:
        """
        将骨骼序列转换为标准化格式
        
        Returns:
            骨骼序列的标准化格式，每个元素为包含关键点坐标的字典
        """
        return [frame.to_normalized_format() for frame in self.frames]
        
    def update_from_normalized(self, normalized_sequence: List[Dict[str, np.ndarray]]):
        """
        从标准化数据更新骨骼序列
        
        Args:
            normalized_sequence: 包含标准化后骨骼序列数据的列表
        """
        if len(normalized_sequence) != len(self.frames):
            raise ValueError("标准化序列长度与原序列不匹配")
            
        for frame, normalized_data in zip(self.frames, normalized_sequence):
            frame.update_from_normalized(normalized_data)
            
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            'fps': float(self.fps) if self.fps is not None else None,
            'duration': float(self.duration) if self.duration is not None else None,
            'frames': [frame.to_dict() for frame in self.frames],
            'metadata': {k: int(v) if isinstance(v, np.integer) else v for k, v in self.metadata.items()}
        }
        
    def save_to_file(self, file_path: Union[str, Path]):
        """保存到文件"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'SkeletonSequence':
        """从文件加载"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        sequence = cls()
        sequence.fps = data['fps']
        sequence.duration = data['duration']
        sequence.metadata = data['metadata']
        
        for frame_data in data['frames']:
            sequence.add_frame(Skeleton.from_dict(frame_data))
            
        return sequence
