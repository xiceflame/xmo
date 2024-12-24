# Pose Skeleton 模块

Pose Skeleton 是 AiYoga 项目的核心模块之一，负责处理和管理姿势骨骼数据。该模块提供了一套完整的工具，用于加载、处理、转换和分析人体姿势骨骼数据。

## 目录结构

```
pose_skeleton/
├── data_structures.py  # 核心数据结构：KeyPoint, Connection, Skeleton, SkeletonSequence
├── data_format.py      # 数据格式转换工具
├── detector.py         # 姿势检测器接口
├── expert_loader.py    # 专家数据加载器
├── loader.py          # 基础数据加载器
├── source.py          # 数据源抽象
├── student_loader.py  # 学员数据加载器
└── tests/             # 单元测试
```

## 核心组件

### 1. 数据结构 (data_structures.py)

#### KeyPoint 类
- **功能**：表示骨骼上的关键点
- **属性**：
  - `id`: 关键点ID
  - `name`: 关键点名称
  - `position`: 3D坐标 (x, y, z)
  - `confidence`: 置信度
- **主要方法**：
  - `distance_to()`: 计算与另一关键点的距离
  - `midpoint_with()`: 计算与另一关键点的中点

#### Connection 类
- **功能**：表示骨骼连接
- **属性**：
  - `start`: 起始关键点ID
  - `end`: 终止关键点ID
  - `type`: 连接类型（如"torso", "left_arm"等）

#### Skeleton 类
- **功能**：表示单帧的完整骨骼
- **属性**：
  - `keypoints`: 关键点字典
  - `connections`: 连接列表
  - `metadata`: 元数据
- **主要方法**：
  - `add_keypoint()`: 添加关键点
  - `add_connection()`: 添加连接
  - `calculate_bone_length()`: 计算骨骼长度
  - `calculate_joint_angle()`: 计算关节角度
  - `calculate_pose_center()`: 计算姿势中心

#### SkeletonSequence 类
- **功能**：表示骨骼序列（多帧）
- **属性**：
  - `frames`: 骨骼帧列表
  - `fps`: 帧率
- **主要方法**：
  - `add_frame()`: 添加骨骼帧
  - `get_frame_slice()`: 获取帧片段
  - `interpolate()`: 插值生成新帧

### 2. 数据格式转换 (data_format.py)

#### DataFormat 类
- **功能**：处理不同格式之间的数据转换
- **支持格式**：
  - 长格式DataFrame (frame, landmark_index, x, y, z)
  - 宽格式DataFrame (x0, y0, z0, x1, y1, z1, ...)
  - Skeleton对象
  - SkeletonSequence对象
- **主要方法**：
  - `long_to_wide()`: 长格式转宽格式
  - `wide_to_long()`: 宽格式转长格式
  - `df_to_skeleton()`: DataFrame转Skeleton
  - `skeleton_to_df()`: Skeleton转DataFrame
  - `df_to_sequence()`: DataFrame转SkeletonSequence
  - `sequence_to_df()`: SkeletonSequence转DataFrame

### 3. 数据加载 (loader.py, expert_loader.py, student_loader.py)

#### BaseLoader 类 (loader.py)
- **功能**：定义数据加载的基本接口
- **主要方法**：
  - `load_from_file()`: 从文件加载
  - `load_from_video()`: 从视频加载
  - `load_from_stream()`: 从流加载

#### ExpertLoader 类 (expert_loader.py)
- **功能**：加载和处理专家姿势数据
- **支持格式**：
  - Parquet文件（长格式）
  - 视频文件

#### StudentLoader 类 (student_loader.py)
- **功能**：加载和处理学员姿势数据
- **支持格式**：
  - 实时视频流
  - 视频文件

### 4. 数据源 (source.py)

#### Source 类
- **功能**：抽象数据源接口
- **子类**：
  - `FileSource`: 文件数据源
  - `VideoSource`: 视频数据源
  - `StreamSource`: 流数据源

### 5. 姿势检测 (detector.py)

#### PoseDetector 类
- **功能**：定义姿势检测接口
- **主要方法**：
  - `detect()`: 检测单帧姿势
  - `detect_sequence()`: 检测视频序列

## 数据流向

```
输入数据 → Source → Loader → Detector → DataFormat → Skeleton/SkeletonSequence
```

## 使用示例

### 1. 加载专家数据

```python
from pose_skeleton.expert_loader import ExpertLoader
from pose_skeleton.data_format import DataFormat

# 初始化加载器和格式转换器
loader = ExpertLoader()
data_format = DataFormat()

# 从Parquet文件加载
sequence = loader.load_from_file("poses/warrior-2/2843.parquet")

# 获取单帧骨骼数据
skeleton = sequence.frames[0]

# 计算关节角度
angle = skeleton.calculate_joint_angle(
    'left_shoulder',
    'left_elbow',
    'left_wrist'
)
```

### 2. 处理学员数据

```python
from pose_skeleton.student_loader import StudentLoader
from pose_skeleton.source import VideoSource

# 初始化加载器和视频源
loader = StudentLoader()
source = VideoSource("webcam")

# 实时处理
for frame in source:
    skeleton = loader.process_frame(frame)
    # 进行姿势分析和比较
```

## 数据标准化

本模块提供了与`pose_normalization`模块的集成接口，用于对骨骼数据进行标准化处理。

### 数据流程

1. 数据采集和处理
   - 通过`pose_skeleton`模块处理原始数据
   - 生成`Skeleton`/`SkeletonSequence`数据结构

2. 数据标准化
   - 使用数据结构类的`to_normalized_format()`方法转换为标准格式
   - 调用`pose_normalization`模块进行标准化
   - 通过`update_from_normalized()`方法更新骨骼数据

### 标准化API

#### Skeleton类
```python
def to_normalized_format(self) -> Dict[str, np.ndarray]:
    """将骨骼数据转换为标准化格式"""
    pass

def update_from_normalized(self, normalized_data: Dict[str, np.ndarray]):
    """从标准化数据更新骨骼"""
    pass
```

#### SkeletonSequence类
```python
def to_normalized_format(self) -> List[Dict[str, np.ndarray]]:
    """将骨骼序列转换为标准化格式"""
    pass

def update_from_normalized(self, normalized_sequence: List[Dict[str, np.ndarray]]):
    """从标准化数据更新骨骼序列"""
    pass
```

### 使用示例

```python
from pose_normalization import PoseNormalizer
from pose_skeleton.data_structures import Skeleton, SkeletonSequence

# 初始化标准化器
normalizer = PoseNormalizer()

# 标准化单帧骨骼数据
skeleton = Skeleton(...)
pose_data = skeleton.to_normalized_format()
normalized_data, confidence = normalizer.normalize_pose_data(pose_data)
skeleton.update_from_normalized(normalized_data)

# 标准化骨骼序列
sequence = SkeletonSequence(...)
sequence_data = sequence.to_normalized_format()
normalized_sequence = normalizer.normalize_pose_sequence_data(sequence_data)
sequence.update_from_normalized(normalized_sequence)
```

## 依赖关系

- **内部依赖**：
  - `data_structures.py` ← 被所有其他模块依赖
  - `data_format.py` ← 被 loader 模块依赖
  - `source.py` ← 被 loader 模块依赖
  - `detector.py` ← 被 loader 模块依赖

- **外部依赖**：
  - NumPy：数值计算
  - Pandas：数据处理
  - OpenCV：视频处理
  - MediaPipe：姿势检测

## 注意事项

1. **数据格式**：
   - 长格式DataFrame必须包含 frame, landmark_index, x, y, z 列
   - 关键点ID必须与MediaPipe格式一致（0-32）

2. **性能考虑**：
   - SkeletonSequence适合处理较短的视频片段
   - 长视频建议使用流式处理

3. **扩展性**：
   - 可以通过继承PoseDetector实现自定义检测器
   - 可以通过继承Source实现自定义数据源

## 未来计划

1. 添加更多边界条件测试
2. 实现数据可视化功能
3. 集成pose_normalization模块
4. 优化实时处理性能
5. 添加数据导出功能
