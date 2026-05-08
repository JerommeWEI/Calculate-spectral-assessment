# FPI 光谱重建评估工具

## 项目简介

本项目提供了一个用于评估法布里-珀罗干涉仪 (FPI) 膜系设计的光谱重建能力的 Python 工具。通过分析传感矩阵的数学特性，评估不同膜系设计对光谱重建精度的影响。

## 功能特性

- **矩阵适定性分析**：可视化传感矩阵热力图、波长互相关矩阵、奇异值衰减曲线
- **信号重建仿真**：支持稀疏信号（单点激光）和连续光谱的重建测试
- **噪声鲁棒性测试**：可调节信噪比 (SNR) 进行噪声环境下的重建评估

## 依赖环境

```
numpy
matplotlib
scikit-learn
```

## 安装

```bash
pip install numpy matplotlib scikit-learn
```

## 使用方法

### 1. 基本用法

```python
from model.assess import FPICoatingEvaluator
import numpy as np

# 定义波长范围
wavelengths = np.linspace(400, 950, 551)

# 加载传感矩阵 (从 Essential Macleod 导出)
# Phi 形状: (N_states, M_wavelengths)
Phi = load_your_sensing_matrix()

# 实例化评估器
evaluator = FPICoatingEvaluator(wavelengths, Phi)

# 分析矩阵特性
evaluator.analyze_matrix()

# 测试稀疏信号重建
evaluator.simulate_reconstruction(signal_type='sparse', snr_db=30)

# 测试连续光谱重建
evaluator.simulate_reconstruction(signal_type='continuous', snr_db=40)
```

### 2. 运行演示

```bash
python model/assess.py
```

## API 参考

### FPICoatingEvaluator 类

#### `__init__(self, wavelengths, sensing_matrix)`

初始化评估器。

| 参数 | 类型 | 说明 |
|------|------|------|
| wavelengths | 1D array | 波长数组 (如 400-950 nm) |
| sensing_matrix | 2D array | 传感矩阵 Phi (N_states × M_wavelengths) |

#### `analyze_matrix(self)`

分析并可视化传感矩阵的数学适定性，生成三个子图：
- 传感矩阵热力图
- 波长互相关矩阵
- 奇异值衰减曲线

#### `simulate_reconstruction(self, signal_type='sparse', snr_db=30)`

执行光谱重建仿真测试。

| 参数 | 类型 | 说明 |
|------|------|------|
| signal_type | str | 信号类型：`'sparse'` (稀疏) 或 `'continuous'` (连续) |
| snr_db | float | 信噪比 (dB) |

## 重建算法

- **稀疏信号**：使用 Lasso (L1 正则化) 回归
- **连续光谱**：使用 Ridge (L2 正则化) 回归

## 数据导入

实际使用时，需从 Essential Macleod 软件导出不同腔长状态下的透射率数据，构建传感矩阵 `Phi`。示例代码中的 Airy 公式仅用于演示目的。

## 项目结构

```
Calculate spectral assessment/
├── model/
│   ├── assess.py                    # 核心评估模块
│   └── visualize_sensing_matrix.py  # 传感矩阵可视化脚本
├── result/                          # 输出图片保存目录
└── README.md                        # 项目说明文档
```

## 版本历史

### v1.1.0 (2026-05-08)

**新增功能**
- 新增 `visualize_sensing_matrix.py` 脚本，支持从目录批量加载 FPI 透射率数据并可视化传感矩阵
- 支持命令行参数指定输入目录
- 自动保存可视化结果到 `result/` 目录，文件名格式：`{时间戳}_{目录名}.png`

**兼容性改进**
- 兼容两种文件命名格式：`*nm.txt` 和 `*nm_*.txt`（如 `0000nm_bpf.txt`）
- 自动检测数据列数，兼容 2 列格式（Wavelength + Transmittance）和 4 列格式（Wavelength + Reflectance + Transmittance + Density）

### v1.0.0 (初始版本)

- 实现 `FPICoatingEvaluator` 类
- 支持矩阵适定性分析（热力图、互相关矩阵、奇异值衰减）
- 支持稀疏信号和连续光谱的重建仿真

## 许可证

MIT License
