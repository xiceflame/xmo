# XMO Pipeline 数据格式指南

本文档详细说明了 XMO Pipeline 支持的各种数据格式的技术细节、版本支持和转换规范。

## 目录
- [通用数据结构](#通用数据结构)
- [BVH 格式](#bvh-格式)
- [FBX 格式](#fbx-格式)
- [JSON 格式](#json-格式)
- [CSV 格式](#csv-格式)
- [MediaPipe 格式](#mediapipe-格式)

## 通用数据结构

在进行格式转换时，所有外部格式都会被转换为系统内部的统一数据结构：

### SkeletonSequence

```python
class SkeletonSequence:
    frames: List[Skeleton]  # 骨骼帧序列
    fps: float             # 帧率
    metadata: Dict         # 元数据
```

### Skeleton

```python
class Skeleton:
    keypoints: Dict[int, KeyPoint]  # 关键点集合
    connections: List[Connection]    # 骨骼连接
    metadata: Dict                  # 元数据
```

### KeyPoint

```python
class KeyPoint:
    landmark_id: int       # 关键点ID
    name: str             # 关键点名称
    position: np.ndarray  # 3D位置 [x, y, z]
    confidence: float     # 置信度 [0.0-1.0]
```

## BVH 格式

### 版本支持
- BioVision Hierarchy (BVH) 标准格式
- 支持版本：1.0及以上

### 技术规范

#### 文件结构
```
HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
    JOINT Chest
    {
        OFFSET 0.00 5.21 0.00
        CHANNELS 3 Zrotation Yrotation Xrotation
        ...
    }
    ...
}
MOTION
Frames: 120
Frame Time: 0.033333
0.00 0.00 0.00 0.00 0.00 0.00 ...
...
```

#### 关节映射
```python
joint_mapping = {
    'Hips': 'pelvis',
    'Spine': 'spine',
    'Spine1': 'spine1',
    'Neck': 'neck',
    'Head': 'head',
    'LeftShoulder': 'left_shoulder',
    # ... 更多映射关系
}
```

#### 转换规则
1. 层级结构转换
   - BVH的层级结构转换为扁平的关键点列表
   - 保持原始的父子关系作为连接信息

2. 坐标系转换
   - BVH使用右手坐标系
   - Y轴向上，Z轴向前，X轴向右
   - 需要进行坐标系对齐

3. 动画数据转换
   - 将每帧的通道数据转换为全局坐标
   - 计算每个关节的全局位置

## FBX 格式

### 版本支持
- Autodesk FBX SDK 2020及以上
- 支持ASCII和二进制格式
- 支持FBX 7.4及以上

### 技术规范

#### SDK依赖
```python
import FbxCommon
# 需要安装 FBX Python SDK
```

#### 节点结构
```
Scene
└── RootNode
    ├── Skeleton
    │   ├── Hips
    │   ├── Spine
    │   └── ...
    └── Animation
        ├── AnimStack
        └── AnimLayer
```

#### 数据访问
```python
# 获取全局变换
global_transform = node.EvaluateGlobalTransform(time)
translation = global_transform.GetT()
rotation = global_transform.GetR()
```

#### 转换规则
1. 骨骼结构转换
   - 遍历FBX节点树
   - 提取骨骼节点信息
   - 保持原始的变换数据

2. 动画数据转换
   - 支持关键帧动画
   - 支持线性插值
   - 保持原始的帧率信息

## JSON 格式

### 标准格式

#### 版本：1.0

#### 文件结构
```json
{
    "format": "standard",
    "version": "1.0",
    "fps": 30.0,
    "frames": [
        {
            "frame_id": 0,
            "keypoints": [
                {
                    "id": 0,
                    "name": "pelvis",
                    "position": [0.0, 0.0, 0.0],
                    "confidence": 1.0
                },
                // ... 更多关键点
            ]
        },
        // ... 更多帧
    ]
}
```

### MediaPipe格式

#### 版本支持
- MediaPipe Pose Solution 0.8.9及以上

#### 数据结构
```json
{
    "pose_landmarks": [
        {
            "landmark": [
                {
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.0,
                    "visibility": 0.9
                },
                // ... 33个关键点
            ]
        },
        // ... 更多帧
    ]
}
```

### OpenPose格式

#### 版本支持
- OpenPose 1.7.0及以上

#### 数据结构
```json
{
    "people": [
        {
            "pose_keypoints_2d": [
                x1, y1, c1,
                x2, y2, c2,
                // ... 25个关键点
            ]
        }
    ]
}
```

## CSV 格式

### 宽格式

#### 文件结构
```csv
frame,x0,y0,z0,x1,y1,z1,...
0,0.1,0.2,0.0,0.3,0.4,0.0,...
1,0.2,0.3,0.0,0.4,0.5,0.0,...
```

#### 规范
- 第一列必须是frame（帧ID）
- 后续列按关键点ID排序
- 每个关键点有x,y,z三个坐标
- 可选confidence列（格式：c0,c1,...）

### 长格式

#### 文件结构
```csv
frame,landmark_index,name,x,y,z,confidence
0,0,pelvis,0.1,0.2,0.0,1.0
0,1,spine,0.2,0.3,0.0,1.0
```

#### 规范
- 必需列：frame, landmark_index, x, y, z
- 可选列：name, confidence
- 按frame和landmark_index排序

## MediaPipe 格式

### 版本支持
- MediaPipe Pose Solution 0.8.9及以上
- Python API输出格式

### 数据结构

#### 关键点映射
```python
landmarks = {
    'nose': 0,
    'left_eye_inner': 1,
    # ... 33个关键点
}
```

#### 输入格式
```python
{
    'landmark': [
        {
            'x': float,  # 归一化坐标 [0.0-1.0]
            'y': float,
            'z': float,
            'visibility': float  # 置信度 [0.0-1.0]
        },
        # ... 33个关键点
    ]
}
```

### 坐标系
- 使用归一化坐标系
- x: 图像宽度方向 [0.0-1.0]
- y: 图像高度方向 [0.0-1.0]
- z: 深度方向，相对于臀部深度

## 格式转换流程

### 导入流程
1. 格式检测
   - 检查文件扩展名
   - 验证文件格式
   - 选择合适的转换器

2. 数据验证
   - 检查必需字段
   - 验证数据类型
   - 检查数值范围

3. 转换处理
   - 读取源数据
   - 进行格式转换
   - 生成目标格式

### 导出流程
1. 预处理
   - 检查数据完整性
   - 准备转换参数
   - 创建输出缓冲

2. 格式转换
   - 坐标系转换
   - 数据结构转换
   - 元数据处理

3. 后处理
   - 格式化输出
   - 数据压缩（如果需要）
   - 文件写入

## 最佳实践

### 数据准备
1. 确保输入数据完整性
2. 检查坐标系一致性
3. 验证帧率信息

### 格式选择
1. BVH：适用于标准动作捕捉数据
2. FBX：适用于3D动画工具链
3. JSON：适用于Web应用和数据交换
4. CSV：适用于数据分析和处理
5. MediaPipe：适用于实时姿势检测

### 性能优化
1. 使用适当的数据类型
2. 避免不必要的格式转换
3. 考虑数据压缩
4. 使用批处理模式
