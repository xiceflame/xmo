# 模块依赖关系分析

## 核心模块

### 1. PoseSkeleton
#### 1.1 基础组件
- **source.py**
  * 功能：提供统一的数据源接口
  * 实现：VideoSource, CameraSource, ParquetSource
  * 依赖：OpenCV, pandas

- **detector.py**
  * 功能：提供统一的姿势检测接口
  * 实现：MediaPipePoseDetector, OpenPosePoseDetector(待实现)
  * 依赖：MediaPipe, OpenPose(待添加)

- **loader.py**
  * 功能：提供统一的数据加载接口
  * 实现：BasePoseLoader, RealtimePoseLoader, BatchPoseLoader
  * 依赖：numpy, pandas

#### 1.2 专用加载器
- **student_loader.py**
  * 功能：学生姿势数据加载
  * 实现：StudentLoader
  * 依赖：
    - source.py (CameraSource, VideoSource)
    - detector.py (MediaPipePoseDetector)
    - loader.py (RealtimePoseLoader, BatchPoseLoader)

- **expert_loader.py**
  * 功能：专家姿势数据加载
  * 实现：ExpertLoader
  * 依赖：
    - source.py (VideoSource, ParquetSource)
    - detector.py (MediaPipePoseDetector)
    - loader.py (BatchPoseLoader)

### 2. PoseNormalization
- **normalizer.py**
  * 功能：姿势标准化处理
  * 主要功能：
    - 位置标准化
    - 旋转标准化
    - 深度标准化
    - 尺度标准化
  * 依赖：numpy, pandas, scipy

### 3. PoseAnalyzer（待实现）
- 功能：分析和比较姿势
- 输入：专家SkeletonSequence和学生SkeletonSequence
- 主要功能：
  * 姿势相似度计算
  * 关键点偏差分析
  * 时序对齐
- 依赖：
  * PoseSkeleton
  * PoseNormalization
  * numpy, scipy

## 模块间依赖关系

1. **数据流向**：
   ```
   数据源(source.py) -> 检测器(detector.py) -> 加载器(loader.py) -> 标准化(normalizer.py) -> 分析器(analyzer.py)
   ```

2. **组件复用**：
   - StudentLoader和ExpertLoader共享基础组件
   - 所有数据经过相同的标准化流程
   - 分析器使用标准化后的数据

## 实现进度

1. **已完成**：
   - PoseSkeleton基础组件
   - 专用加载器
   - PoseNormalization

2. **进行中**：
   - OpenPosePoseDetector实现
   - 数据格式优化
   - 性能优化

3. **待实现**：
   - PoseAnalyzer
   - 更多数据源支持
   - 高级分析功能

## 注意事项

1. **代码质量**：
   - 遵循SOLID原则
   - 保持接口一致性
   - 编写完整的单元测试

2. **性能优化**：
   - 使用异步IO
   - 实现数据缓存
   - 支持批处理
   - 考虑GPU加速

3. **可扩展性**：
   - 保持接口抽象
   - 支持插件式架构
   - 预留扩展点
