"""
姿势标准化模块

本模块负责对姿势数据进行标准化处理，包括：
1. 位置标准化：将髋部中心点移至坐标原点
2. 旋转标准化：基于肩部线对齐身体朝向
3. 深度标准化：确保身体正面朝向摄像机
4. 尺度标准化：使用躯干长度作为参考单位

主要功能：
- 统一使用MediaPipe的33个关键点作为基础数据结构
- 采用相对坐标系统，以髋部中心为参考点
- 实现深度信息的Z-score标准化
- 保持左右对称性的骨骼长度归一化

技术实现：
1. 位置标准化
   - 计算髋部中心点
   - 将所有点相对于髋部中心点平移

2. 旋转标准化
   - 使用肩部线作为参考
   - 计算旋转矩阵对齐到标准方向

3. 深度标准化
   - 计算躯干平面法向量
   - 旋转使法向量对齐到z轴

4. 尺度标准化
   - 计算躯干长度作为参考
   - 对所有坐标进行归一化

使用方法：
```python
normalizer = PoseNormalizer()
normalized_pose = normalizer.normalize_pose(pose_data)
```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation

class PoseNormalizer:
    """姿势标准化器，用于处理和标准化3D姿势数据"""
    
    def __init__(self):
        """初始化姿势标准化器"""
        # MediaPipe关键点映射
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
        
        # 定义标准骨骼长度比例（基于人体工程学数据）
        self.standard_bone_ratios = {
            'shoulder_width': 1.0,  # 作为基准
            'upper_arm': 0.818,     # 相对于肩宽
            'forearm': 0.723,
            'hip_width': 0.818,
            'upper_leg': 1.205,
            'lower_leg': 1.116
        }
        
        # 定义骨骼连接
        self.bones = {
            'torso': [
                ('left_shoulder', 'right_shoulder'),
                ('left_hip', 'right_hip'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip')
            ],
            'left_arm': [
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist')
            ],
            'right_arm': [
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist')
            ],
            'left_leg': [
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle')
            ],
            'right_leg': [
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle')
            ]
        }

    def get_coordinates(self, frame_data: pd.Series, landmark_idx: int) -> np.ndarray:
        """
        从帧数据中获取特定关键点的坐标
        
        Args:
            frame_data: 包含坐标数据的Series
            landmark_idx: 关键点索引
            
        Returns:
            包含(x,y,z)坐标的numpy数组
        """
        try:
            # 计算起始索引
            start_idx = landmark_idx * 3
            
            # 获取坐标
            coords = np.array([
                frame_data[start_idx],
                frame_data[start_idx + 1],
                frame_data[start_idx + 2]
            ])
            
            return coords
        except Exception as e:
            print(f"Warning: Failed to get coordinates for landmark {landmark_idx}: {str(e)}")
            return np.zeros(3)
    
    def calculate_mid_point(self, frame_data: pd.Series, point1_idx: int, point2_idx: int) -> np.ndarray:
        """计算两个关键点的中点"""
        point1 = self.get_coordinates(frame_data, point1_idx)
        point2 = self.get_coordinates(frame_data, point2_idx)
        return (point1 + point2) / 2

    def normalize_position(self, frame_data: pd.Series) -> pd.Series:
        """
        标准化位置
        将髋部中心点移动到原点
        """
        try:
            # 计算髋部中心
            mid_hip = self.calculate_mid_point(
                frame_data,
                self.landmarks['left_hip'],
                self.landmarks['right_hip']
            )
            
            # 检查mid_hip是否为NaN
            if np.any(np.isnan(mid_hip)):
                print("Warning: mid_hip contains NaN values")
                return frame_data
            
            # 平移所有点使mid_hip位于原点
            normalized_data = frame_data.copy()
            for i in range(0, len(frame_data), 3):
                normalized_data[i:i+3] = frame_data[i:i+3] - mid_hip
                
            return normalized_data
        except Exception as e:
            print(f"Warning: Failed to normalize position: {str(e)}")
            return frame_data

    def normalize_rotation(self, frame_data: pd.Series) -> pd.Series:
        """
        标准化身体朝向
        使用肩部线作为参考，将其旋转至标准方向
        """
        try:
            # 获取左右肩膀的坐标
            left_shoulder = self.get_coordinates(frame_data, self.landmarks['left_shoulder'])
            right_shoulder = self.get_coordinates(frame_data, self.landmarks['right_shoulder'])
            
            # 计算肩部向量
            shoulder_vector = right_shoulder - left_shoulder
            
            # 检查向量是否为零向量
            if np.allclose(shoulder_vector, 0):
                print("Warning: shoulder vector is zero")
                return frame_data
                
            # 计算向量的范数
            norm = np.linalg.norm(shoulder_vector)
            if norm < 1e-6:  # 如果向量太小
                print("Warning: shoulder vector is too small")
                return frame_data
                
            # 目标方向（我们希望肩部线与x轴平行）
            target_vector = np.array([1.0, 0.0, 0.0])
            
            # 计算旋转矩阵
            rotation_matrix = Rotation.align_vectors(
                target_vector.reshape(1, -1),
                shoulder_vector.reshape(1, -1)
            )[0].as_matrix()
            
            # 应用旋转到所有关键点
            normalized_frame = frame_data.copy()
            for landmark_idx in self.landmarks.values():
                point = self.get_coordinates(frame_data, landmark_idx)
                rotated_point = np.dot(rotation_matrix, point)
                normalized_frame[landmark_idx*3:(landmark_idx+1)*3] = rotated_point
                    
            return normalized_frame
            
        except Exception as e:
            print(f"Warning: Failed to normalize rotation: {str(e)}")
            return frame_data

    def normalize_depth(self, frame_data: pd.Series) -> pd.Series:
        """
        标准化深度
        确保身体正面朝向摄像机（z轴）
        """
        try:
            # 计算躯干法向量
            left_shoulder = self.get_coordinates(frame_data, self.landmarks['left_shoulder'])
            right_shoulder = self.get_coordinates(frame_data, self.landmarks['right_shoulder'])
            left_hip = self.get_coordinates(frame_data, self.landmarks['left_hip'])
            
            # 计算躯干平面法向量
            v1 = left_shoulder - right_shoulder
            v2 = left_hip - right_shoulder
            
            # 检查向量是否为零向量
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                return frame_data
                
            normal = np.cross(v1, v2)
            
            # 检查法向量是否为零向量
            if np.allclose(normal, 0):
                return frame_data
                
            normal = normal / np.linalg.norm(normal)
            
            # 计算将法向量对齐到z轴的旋转
            target_normal = np.array([0, 0, 1])
            rotation_matrix = Rotation.align_vectors(
                target_normal.reshape(1, -1),
                normal.reshape(1, -1)
            )[0].as_matrix()
            
            # 应用旋转
            aligned_data = frame_data.copy()
            for i in range(0, len(frame_data), 3):
                point = frame_data[i:i+3]
                aligned_point = rotation_matrix @ point
                aligned_data[i:i+3] = aligned_point
                
            return aligned_data
        except Exception as e:
            print(f"Warning: Failed to normalize depth: {str(e)}")
            return frame_data

    def normalize_scale(self, frame_data: pd.Series) -> pd.Series:
        """
        基于人体高度进行尺度标准化
        使用躯干长度作为参考
        """
        try:
            # 计算躯干长度
            shoulder_center = self.calculate_mid_point(
                frame_data,
                self.landmarks['left_shoulder'],
                self.landmarks['right_shoulder']
            )
            hip_center = self.calculate_mid_point(
                frame_data,
                self.landmarks['left_hip'],
                self.landmarks['right_hip']
            )
            
            # 检查坐标是否为NaN
            if np.any(np.isnan(shoulder_center)) or np.any(np.isnan(hip_center)):
                print("Warning: shoulder_center or hip_center contains NaN values")
                return frame_data
            
            trunk_length = np.linalg.norm(shoulder_center - hip_center)
            
            # 检查躯干长度是否接近零
            if trunk_length < 1e-6:
                print("Warning: trunk length is too small")
                return frame_data
            
            # 标准化所有坐标
            normalized_data = frame_data.copy()
            for i in range(0, len(frame_data), 3):
                normalized_data[i:i+3] = frame_data[i:i+3] / trunk_length
                
            return normalized_data
        except Exception as e:
            print(f"Warning: Failed to normalize scale: {str(e)}")
            return frame_data

    def normalize_pose(self, frame_data: pd.Series) -> Tuple[pd.Series, float]:
        """
        对单帧姿势数据进行完整的标准化处理
        
        Returns:
            Tuple[pd.Series, float]: (标准化后的数据, 置信度分数)
        """
        try:
            # 1. 位置标准化
            normalized_data = self.normalize_position(frame_data)
            
            # 2. 旋转标准化
            normalized_data = self.normalize_rotation(normalized_data)
            
            # 3. 深度标准化
            normalized_data = self.normalize_depth(normalized_data)
            
            # 4. 尺度标准化
            normalized_data = self.normalize_scale(normalized_data)
            
            # 计算置信度分数（可以根据需要调整计算方法）
            confidence_score = 1.0
            
            return normalized_data, confidence_score
            
        except Exception as e:
            print(f"Warning: Failed to normalize pose: {str(e)}")
            return frame_data, 0.0

    def normalize_pose_sequence(self, pose_data: pd.DataFrame) -> pd.DataFrame:
        """
        对整个姿势序列进行标准化
        
        Args:
            pose_data: 包含多帧姿势数据的DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        normalized_frames = []
        confidence_scores = []
        
        for _, frame in pose_data.iterrows():
            normalized_frame, confidence = self.normalize_pose(frame)
            normalized_frames.append(normalized_frame)
            confidence_scores.append(confidence)
            
        result_df = pd.DataFrame(normalized_frames)
        result_df['confidence_score'] = confidence_scores
        
        return result_df

    def normalize_pose_data(self, pose_data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """
        标准化姿势数据
        
        Args:
            pose_data: 包含关键点坐标的字典，格式为 {'landmark_name': np.array([x, y, z])}
            
        Returns:
            Tuple[Dict[str, np.ndarray], float]: (标准化后的姿势数据, 置信度分数)
        """
        # 将字典数据转换为Series格式
        frame_data = pd.Series()
        for name, coords in pose_data.items():
            if name in self.landmarks:
                idx = self.landmarks[name]
                frame_data[idx*3:(idx+1)*3] = coords
                
        # 进行标准化
        normalized_series, confidence = self.normalize_pose(frame_data)
        
        # 转换回字典格式
        normalized_data = {}
        for name, idx in self.landmarks.items():
            normalized_data[name] = normalized_series[idx*3:(idx+1)*3].values
            
        return normalized_data, confidence
        
    def normalize_pose_sequence_data(self, pose_sequence: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """
        标准化姿势序列数据
        
        Args:
            pose_sequence: 姿势序列列表，每个元素为包含关键点坐标的字典
            
        Returns:
            标准化后的姿势序列
        """
        # 将序列数据转换为DataFrame格式
        frame_data_list = []
        for pose_data in pose_sequence:
            series = pd.Series()
            for name, coords in pose_data.items():
                if name in self.landmarks:
                    idx = self.landmarks[name]
                    series[idx*3:(idx+1)*3] = coords
            frame_data_list.append(series)
            
        pose_df = pd.DataFrame(frame_data_list)
        
        # 进行标准化
        normalized_df = self.normalize_pose_sequence(pose_df)
        
        # 转换回字典列表格式
        normalized_sequence = []
        for _, row in normalized_df.iterrows():
            normalized_data = {}
            for name, idx in self.landmarks.items():
                normalized_data[name] = row[idx*3:(idx+1)*3].values
            normalized_sequence.append(normalized_data)
            
        return normalized_sequence

    def calculate_bone_length(self, frame_data: pd.Series, start_point: str, end_point: str) -> float:
        """
        计算两个关键点之间的骨骼长度
        
        Args:
            frame_data: 帧数据
            start_point: 起始关键点名称
            end_point: 终止关键点名称
            
        Returns:
            骨骼长度
        """
        try:
            start_coords = self.get_coordinates(frame_data, self.landmarks[start_point])
            end_coords = self.get_coordinates(frame_data, self.landmarks[end_point])
            
            return np.linalg.norm(end_coords - start_coords)
        except Exception as e:
            print(f"Warning: Failed to calculate bone length between {start_point} and {end_point}: {str(e)}")
            return 0.0

    def calculate_joint_angle(self, frame_data: pd.Series, point1: str, point2: str, point3: str) -> float:
        """
        计算三个关键点形成的角度
        
        Args:
            frame_data: 帧数据
            point1: 第一个关键点名称
            point2: 中心关键点名称（角度顶点）
            point3: 第三个关键点名称
            
        Returns:
            角度（弧度）
        """
        try:
            p1 = self.get_coordinates(frame_data, self.landmarks[point1])
            p2 = self.get_coordinates(frame_data, self.landmarks[point2])
            p3 = self.get_coordinates(frame_data, self.landmarks[point3])
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # 检查向量是否为零向量
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                return 0.0
                
            # 计算角度
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # 处理数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            return np.arccos(cos_angle)
        except Exception as e:
            print(f"Warning: Failed to calculate joint angle for {point1}-{point2}-{point3}: {str(e)}")
            return 0.0

    def verify_bone_lengths(self, frame_data: pd.Series) -> Dict[str, float]:
        """
        验证骨骼长度的一致性
        
        Returns:
            包含各骨骼长度比例的字典
        """
        try:
            # 计算肩宽作为基准
            shoulder_width = self.calculate_bone_length(
                frame_data,
                'left_shoulder',
                'right_shoulder'
            )
            
            if shoulder_width < 1e-6:
                return {}
                
            # 计算各部分骨骼长度比例
            ratios = {}
            
            # 上臂
            left_upper_arm = self.calculate_bone_length(frame_data, 'left_shoulder', 'left_elbow')
            right_upper_arm = self.calculate_bone_length(frame_data, 'right_shoulder', 'right_elbow')
            ratios['left_upper_arm'] = left_upper_arm / shoulder_width
            ratios['right_upper_arm'] = right_upper_arm / shoulder_width
            
            # 前臂
            left_forearm = self.calculate_bone_length(frame_data, 'left_elbow', 'left_wrist')
            right_forearm = self.calculate_bone_length(frame_data, 'right_elbow', 'right_wrist')
            ratios['left_forearm'] = left_forearm / shoulder_width
            ratios['right_forearm'] = right_forearm / shoulder_width
            
            # 大腿
            left_thigh = self.calculate_bone_length(frame_data, 'left_hip', 'left_knee')
            right_thigh = self.calculate_bone_length(frame_data, 'right_hip', 'right_knee')
            ratios['left_thigh'] = left_thigh / shoulder_width
            ratios['right_thigh'] = right_thigh / shoulder_width
            
            # 小腿
            left_shin = self.calculate_bone_length(frame_data, 'left_knee', 'left_ankle')
            right_shin = self.calculate_bone_length(frame_data, 'right_knee', 'right_ankle')
            ratios['left_shin'] = left_shin / shoulder_width
            ratios['right_shin'] = right_shin / shoulder_width
            
            return ratios
        except Exception as e:
            print(f"Warning: Failed to verify bone lengths: {str(e)}")
            return {}

    def extract_pose_features(self, frame_data: pd.Series) -> Dict[str, float]:
        """
        提取姿势特征，包括关键关节角度和相对位置
        
        Returns:
            包含姿势特征的字典
        """
        try:
            features = {}
            
            # 计算关键关节角度
            # 肘部角度
            features['left_elbow_angle'] = self.calculate_joint_angle(
                frame_data,
                'left_shoulder',
                'left_elbow',
                'left_wrist'
            )
            features['right_elbow_angle'] = self.calculate_joint_angle(
                frame_data,
                'right_shoulder',
                'right_elbow',
                'right_wrist'
            )
            
            # 膝盖角度
            features['left_knee_angle'] = self.calculate_joint_angle(
                frame_data,
                'left_hip',
                'left_knee',
                'left_ankle'
            )
            features['right_knee_angle'] = self.calculate_joint_angle(
                frame_data,
                'right_hip',
                'right_knee',
                'right_ankle'
            )
            
            # 髋部角度
            features['left_hip_angle'] = self.calculate_joint_angle(
                frame_data,
                'left_shoulder',
                'left_hip',
                'left_knee'
            )
            features['right_hip_angle'] = self.calculate_joint_angle(
                frame_data,
                'right_shoulder',
                'right_hip',
                'right_knee'
            )
            
            # 肩部角度
            features['left_shoulder_angle'] = self.calculate_joint_angle(
                frame_data,
                'right_shoulder',
                'left_shoulder',
                'left_elbow'
            )
            features['right_shoulder_angle'] = self.calculate_joint_angle(
                frame_data,
                'left_shoulder',
                'right_shoulder',
                'right_elbow'
            )
            
            return features
        except Exception as e:
            print(f"Warning: Failed to extract pose features: {str(e)}")
            return {}