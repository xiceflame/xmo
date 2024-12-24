# Pose Normalization Module

This module provides standardized 3D pose normalization functionality, designed for research projects in human pose analysis.

## Features

- Position normalization: Centers the pose around hip center
- Rotation normalization: Aligns body orientation using shoulder line
- Depth normalization: Ensures body faces the camera
- Scale normalization: Uses torso length as reference unit
- Bone length verification
- Joint angle calculation
- Pose feature extraction

## Installation

```bash
pip install -e .
```

## Usage

```python
from pose_normalization import PoseNormalizer

# Initialize normalizer
normalizer = PoseNormalizer()

# Normalize a single pose frame
normalized_pose, confidence = normalizer.normalize_pose(pose_data)

# Process a sequence of poses
normalized_sequence = normalizer.normalize_pose_sequence(pose_sequence)

# Extract pose features
features = normalizer.extract_pose_features(pose_data)
```

## Input Data Format

The module expects pose data in the following format:
- Each frame should be a pandas Series with 99 values (33 landmarks × 3 coordinates)
- The coordinates should be in the order: [x1, y1, z1, x2, y2, z2, ...]
- The landmarks follow the MediaPipe pose landmark convention

## Dependencies

- numpy
- pandas
- scipy
- typing

## Testing

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Citation

If you use this module in your research, please cite:

```bibtex
@misc{pose_normalization,
  author = {Your Name},
  title = {Pose Normalization Module},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/pose_normalization}
}
```
