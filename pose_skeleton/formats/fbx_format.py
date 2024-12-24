"""
FBX格式转换器

处理FBX（Filmbox）动作捕捉数据格式
"""

from typing import Dict, List, Optional
import numpy as np
from .base_format import BaseFormat
from ..data_structures import SkeletonSequence, Skeleton, KeyPoint

class FBXFormat(BaseFormat):
    """FBX格式转换器"""
    
    def __init__(self):
        """初始化FBX格式转换器"""
        self.joint_mapping: Dict[str, str] = {
            # FBX关节到标准关节的映射
            'Hips': 'pelvis',
            'Spine': 'spine',
            'Spine1': 'spine1',
            'Neck': 'neck',
            'Head': 'head',
            'LeftShoulder': 'left_shoulder',
            'LeftArm': 'left_upper_arm',
            'LeftForeArm': 'left_lower_arm',
            'LeftHand': 'left_hand',
            'RightShoulder': 'right_shoulder',
            'RightArm': 'right_upper_arm',
            'RightForeArm': 'right_lower_arm',
            'RightHand': 'right_hand',
            'LeftUpLeg': 'left_upper_leg',
            'LeftLeg': 'left_lower_leg',
            'LeftFoot': 'left_foot',
            'RightUpLeg': 'right_upper_leg',
            'RightLeg': 'right_lower_leg',
            'RightFoot': 'right_foot'
        }
        
        try:
            import FbxCommon
            self._has_fbx = True
        except ImportError:
            self._has_fbx = False
    
    def load(self, source: str) -> SkeletonSequence:
        """
        从FBX文件加载动作数据
        
        Args:
            source: FBX文件路径
            
        Returns:
            SkeletonSequence: 转换后的骨骼序列
        """
        if not self._has_fbx:
            raise ImportError("FBX SDK not found. Please install the FBX Python SDK.")
            
        import FbxCommon
        
        # 初始化FBX SDK管理器和场景
        manager = FbxCommon.FbxManager.Create()
        scene = FbxCommon.FbxScene.Create(manager, "")
        
        # 导入器
        importer = FbxCommon.FbxImporter.Create(manager, "")
        import_status = importer.Initialize(source)
        
        if not import_status:
            raise RuntimeError(f"Failed to load FBX file: {source}")
            
        # 导入场景
        importer.Import(scene)
        importer.Destroy()
        
        # 创建骨骼序列
        sequence = SkeletonSequence()
        
        # 获取动画帧率
        fps = scene.GetGlobalSettings().GetTimeMode().GetFrameRate()
        sequence.fps = float(fps)
        
        # 获取根节点
        root_node = scene.GetRootNode()
        
        # 获取动画时间范围
        start_time = FbxCommon.FbxTime()
        end_time = FbxCommon.FbxTime()
        time_span = scene.GetGlobalSettings().GetTimelineDefaultTimeSpan()
        start_time.SetFrame(0, FbxCommon.FbxTime.eFrames30)
        end_time.SetFrame(time_span.GetDuration().GetFrameCount(), FbxCommon.FbxTime.eFrames30)
        
        # 遍历每一帧
        current_time = start_time
        while current_time <= end_time:
            skeleton = self._process_frame(root_node, current_time)
            sequence.add_frame(skeleton)
            current_time.Set(current_time.Get() + current_time.GetFrameTime().Get())
            
        manager.Destroy()
        return sequence
    
    def save(self, skeleton_sequence: SkeletonSequence, target: str) -> None:
        """
        将骨骼序列保存为FBX格式
        
        Args:
            skeleton_sequence: 要保存的骨骼序列
            target: 目标FBX文件路径
        """
        if not self._has_fbx:
            raise ImportError("FBX SDK not found. Please install the FBX Python SDK.")
            
        import FbxCommon
        
        # 初始化FBX SDK管理器和场景
        manager = FbxCommon.FbxManager.Create()
        scene = FbxCommon.FbxScene.Create(manager, "")
        
        # TODO: 实现FBX文件生成
        # 1. 创建骨骼层级
        # 2. 创建动画曲线
        # 3. 设置关键帧数据
        
        # 导出器
        exporter = FbxCommon.FbxExporter.Create(manager, "")
        export_status = exporter.Initialize(target)
        
        if not export_status:
            raise RuntimeError(f"Failed to initialize FBX exporter for: {target}")
            
        # 导出场景
        exporter.Export(scene)
        exporter.Destroy()
        manager.Destroy()
    
    def validate(self, data: str) -> bool:
        """
        验证FBX文件格式
        
        Args:
            data: FBX文件路径
            
        Returns:
            bool: 文件格式是否有效
        """
        if not self._has_fbx:
            return False
            
        try:
            import FbxCommon
            manager = FbxCommon.FbxManager.Create()
            importer = FbxCommon.FbxImporter.Create(manager, "")
            status = importer.Initialize(data)
            importer.Destroy()
            manager.Destroy()
            return status
        except Exception:
            return False
            
    def _process_frame(self, node: 'FbxNode', time: 'FbxTime') -> Skeleton:
        """
        处理单帧数据
        
        Args:
            node: FBX节点
            time: 当前时间
            
        Returns:
            Skeleton: 处理后的骨骼数据
        """
        skeleton = Skeleton('fbx')
        self._traverse_node(node, time, skeleton)
        return skeleton
        
    def _traverse_node(self, node: 'FbxNode', time: 'FbxTime', skeleton: Skeleton) -> None:
        """
        遍历FBX节点树
        
        Args:
            node: FBX节点
            time: 当前时间
            skeleton: 骨骼对象
        """
        # 获取全局变换
        global_transform = node.EvaluateGlobalTransform(time)
        translation = global_transform.GetT()
        
        # 如果是关节节点且在映射中
        node_name = node.GetName()
        if node_name in self.joint_mapping:
            keypoint = KeyPoint(
                landmark_id=len(skeleton.keypoints),
                name=self.joint_mapping[node_name],
                position=np.array([translation[0], translation[1], translation[2]]),
                confidence=1.0
            )
            skeleton.add_keypoint(keypoint)
        
        # 递归处理子节点
        for i in range(node.GetChildCount()):
            self._traverse_node(node.GetChild(i), time, skeleton)
