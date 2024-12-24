# XMO Pipeline

XMO Pipeline 是一个基于动作捕捉的动作评价系统。该系统通过处理和分析人体动作数据，提供实时的动作评估和反馈。

## 目录
- [系统概述](#系统概述)
- [技术架构](#技术架构)
- [处理流程](#处理流程)
- [数据格式](#数据格式)
- [使用指南](#使用指南)
- [开发指南](#开发指南)
- [部署说明](#部署说明)

## 系统概述

### 核心功能
- 多源动作数据采集和处理
- 实时姿势检测和分析
- 动作标准化和评估
- 详细的评估报告生成

### 主要特点
- 支持多种数据输入源（视频、摄像头、文件）
- 灵活的姿势检测器接口（MediaPipe、OpenPose）
- 完整的姿势标准化处理
- 丰富的数据格式支持（BVH、FBX、JSON、CSV）
- 实时处理和反馈能力
- 批量数据处理支持

## 技术架构

### 核心模块

#### 1. PoseSkeleton
动作捕捉和骨骼数据处理模块

**组件说明：**
- `source.py`: 统一数据源接口
  * 支持视频输入
  * 支持摄像头输入
  * 支持文件输入（Parquet等）

- `detector.py`: 姿势检测接口
  * MediaPipe检测器
  * OpenPose检测器（规划中）
  * 自定义检测器扩展支持

- `data_format.py`: 数据格式管理
  * 统一的格式转换入口
  * 多格式支持和转换
  * 数据验证和错误处理

- `formats/`: 格式转换器集合
  * BVH格式：动作捕捉标准格式
  * FBX格式：3D动画标准格式
  * JSON格式：Web交互标准格式
  * CSV格式：数据分析标准格式
  * MediaPipe格式：实时检测输出格式

#### 2. PoseNormalization
姿势标准化处理模块

**标准化流程：**

1. 位置标准化
   - 坐标系对齐：统一不同数据源的坐标系
   - 原点对齐：将骨骼根节点对齐到原点
   - 尺度归一化：统一骨骼的整体尺度
   - 抖动消除：去除高频噪声

2. 旋转标准化
   - 方向对齐：统一面向方向
   - 姿态校正：修正倾斜和歪斜
   - 角度标准化：统一关节角度范围
   - 旋转插值：平滑旋转变化

3. 深度标准化
   - 深度校准：统一深度基准
   - Z轴对齐：修正前后倾斜
   - 比例调整：统一深度比例
   - 深度平滑：消除深度抖动

4. 尺度标准化
   - 骨骼长度归一化：统一骨骼长度比例
   - 关节角度校正：修正关节限制
   - 比例校正：保持身体各部分比例
   - 全局缩放：适应不同显示需求

**标准化配置：**

```python
class NormalizationConfig:
    # 位置标准化
    position_normalization = {
        'root_align': True,      # 根节点对齐
        'scale_normalize': True, # 尺度归一化
        'smooth_factor': 0.1,   # 平滑因子
    }
    
    # 旋转标准化
    rotation_normalization = {
        'face_forward': True,   # 朝向统一
        'up_vector': [0,1,0],   # 向上向量
        'angle_limits': True,   # 角度限制
    }
    
    # 深度标准化
    depth_normalization = {
        'depth_scale': 1.0,     # 深度缩放
        'z_align': True,        # Z轴对齐
        'depth_smooth': 0.1,    # 深度平滑
    }
    
    # 尺度标准化
    scale_normalization = {
        'bone_lengths': True,   # 骨骼长度
        'keep_proportions': True, # 保持比例
        'global_scale': 1.0,    # 全局缩放
    }
```

**使用示例：**

```python
from pose_normalization import PoseNormalizer, NormalizationConfig

# 创建标准化配置
config = NormalizationConfig()
config.position_normalization['smooth_factor'] = 0.2
config.rotation_normalization['face_forward'] = True

# 初始化标准化器
normalizer = PoseNormalizer(config)

# 处理单帧数据
normalized_frame = normalizer.normalize_frame(frame)

# 处理序列数据
normalized_sequence = normalizer.normalize_sequence(sequence)
```

**标准化效果：**

1. 位置标准化
   - 消除位置偏移
   - 统一运动范围
   - 减少抖动干扰

2. 旋转标准化
   - 统一动作方向
   - 修正姿势倾斜
   - 平滑旋转变化

3. 深度标准化
   - 统一深度范围
   - 修正前后倾斜
   - 改善深度感知

4. 尺度标准化
   - 统一骨骼比例
   - 修正关节限制
   - 适应显示需求

**质量控制：**

1. 输入验证
   - 数据完整性检查
   - 格式兼容性验证
   - 数值范围检查

2. 处理监控
   - 标准化过程日志
   - 异常值检测
   - 性能指标跟踪

3. 结果验证
   - 标准化效果评估
   - 数据一致性检查
   - 视觉质量验证

#### 3. PoseEvaluation
动作评估和分析模块

**评估维度：**
1. 姿势相似度
   - 关键点位置比较
   - 姿态角度比较
   - 整体形态比较

2. 时序对齐
   - 动作速度分析
   - 关键帧对齐
   - 时序特征提取

3. 评分和反馈
   - 分项评分生成
   - 问题检测
   - 改进建议

## 处理流程

### 1. 数据采集
```
视频/摄像头/文件 -> 数据源接口 -> 原始数据
```

**关键步骤：**
- 选择合适的数据源
- 配置采集参数
- 数据预处理

### 2. 姿势检测
```
原始数据 -> 姿势检测器 -> 关键点数据
```

**检测流程：**
1. 图像预处理
2. 人体检测
3. 关键点定位
4. 置信度计算

### 3. 数据转换
```
关键点数据 -> 格式转换器 -> 标准格式
```

**转换规则：**
1. 格式识别和验证
2. 数据结构转换
3. 坐标系对齐
4. 数据完整性检查

### 4. 姿势标准化
```
标准格式 -> 标准化处理 -> 规范化姿势
```

**标准化步骤：**
1. 位置归一化
2. 旋转对齐
3. 深度校准
4. 尺度调整

### 5. 动作评估
```
规范化姿势 -> 评估系统 -> 评估报告
```

**评估流程：**
1. 特征提取
2. 相似度计算
3. 问题检测
4. 报告生成

## 数据格式

### 通用数据结构

#### SkeletonSequence
动作序列的核心数据结构
```python
class SkeletonSequence:
    frames: List[Skeleton]  # 骨骼帧序列
    fps: float             # 帧率
    metadata: Dict         # 元数据
```

#### Skeleton
单帧姿势的数据结构
```python
class Skeleton:
    keypoints: Dict[int, KeyPoint]  # 关键点集合
    connections: List[Connection]    # 骨骼连接
    metadata: Dict                  # 元数据
```

### 支持格式

#### 1. BVH格式
- 版本：1.0及以上
- 用途：动作捕捉数据交换
- 特点：标准骨骼层级结构

#### 2. FBX格式
- 版本：7.4及以上
- 用途：3D动画数据交换
- 特点：专业动画工具标准

#### 3. JSON格式
- 标准格式：自定义规范
- MediaPipe格式：实时检测输出
- OpenPose格式：姿势估计结果

#### 4. CSV格式
- 宽格式：适用于帧序列数据
- 长格式：适用于关键点序列
- 特点：易于数据分析和处理

## 使用指南

### 1. 环境准备
```bash
# 克隆项目
git clone [repository-url]
cd xmo-pipeline

# 安装依赖
pip install -r requirements.txt
```

### 2. 基础用法
```python
from pose_skeleton import DataFormatManager
from pose_normalization import PoseNormalizer
from pose_evaluation import PoseEvaluator

# 初始化组件
format_manager = DataFormatManager()
normalizer = PoseNormalizer()
evaluator = PoseEvaluator()

# 处理动作数据
sequence = format_manager.load_file("motion.bvh")
normalized = normalizer.normalize(sequence)
result = evaluator.evaluate(normalized)

# 保存结果
format_manager.save_file(sequence, "output.json")
```

### 3. 实时处理
```python
from pose_skeleton import StudentLoader, MediaPipePoseDetector

# 初始化实时处理
detector = MediaPipePoseDetector()
loader = StudentLoader(detector)

# 处理视频流
for frame in loader.process_stream(0):  # 0为摄像头ID
    normalized = normalizer.normalize(frame)
    result = evaluator.evaluate(normalized)
    # 处理结果...
```

## 开发指南

### 代码规范
1. 遵循PEP 8编码规范
2. 使用类型注解
3. 编写完整的文档字符串
4. 保持代码简洁清晰

### 测试规范
1. 单元测试覆盖核心功能
2. 集成测试验证模块交互
3. 性能测试确保实时性
4. 定期运行测试套件

### 文档规范
1. 及时更新API文档
2. 维护使用示例
3. 记录重要变更
4. 提供故障排除指南

## 部署说明

### 系统要求
- Python 3.8+
- CUDA支持（推荐）
- 足够的计算资源

### 依赖安装
```bash
# 基础依赖
pip install -r requirements.txt

# 可选依赖（用于FBX支持）
pip install fbx-python
```

### 性能优化
1. 使用GPU加速
2. 启用批处理模式
3. 优化数据加载
4. 调整缓存策略

### 监控和维护
1. 日志记录
2. 性能监控
3. 错误追踪
4. 定期备份

## 开发状态

- [x] PoseSkeleton基础组件
- [x] 专用数据加载器
- [x] PoseNormalization
- [x] 多格式支持
- [ ] OpenPose检测器
- [ ] PoseEvaluation
- [ ] 高级分析功能

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

[License Name] - 详见LICENSE文件
