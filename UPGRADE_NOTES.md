# StrongSORT 升级说明

## 升级内容

本次升级为pip版本的StrongSORT添加了以下功能：

### 1. GSI (Gaussian-smoothed interpolation)
- **功能**: 高斯平滑插值，补偿缺失的检测结果
- **原理**: 使用高斯过程回归算法进行轨迹平滑
- **优势**: 比线性插值更准确，能更好地处理运动信息

### 2. AFLink (Appearance-free link)
- **功能**: 无外观信息的轨迹关联
- **原理**: 基于轨迹几何特征进行全局链接
- **优势**: 可以将短轨迹片段关联成完整轨迹

## 新增依赖

升级后需要安装以下额外依赖：
```bash
pip install scikit-learn>=0.19.2
pip install scipy>=1.7.0
```

## 使用方法

### 基础使用（保持原有API兼容）
```python
from strongsort import StrongSORT

tracker = StrongSORT(
    model_weights='model.pth',
    device='cuda'
)
```

### 启用增强功能
```python
from strongsort import StrongSORT

tracker = StrongSORT(
    model_weights='osnet_x0_25_market1501.pth',
    device='cuda',
    # 新增参数
    enable_aflink=True,           # 启用AFLink
    enable_gsi=True,              # 启用GSI
    aflink_model_path='AFLink_epoch20.pth',  # AFLink模型路径
    gsi_interval=20,              # GSI插值间隔
    gsi_tau=10,                   # GSI平滑参数
)
```

## 新增参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_aflink` | bool | False | 是否启用AFLink功能 |
| `enable_gsi` | bool | False | 是否启用GSI功能 |
| `aflink_model_path` | str | None | AFLink模型文件路径 |
| `gsi_interval` | int | 20 | GSI插值最大间隔 |
| `gsi_tau` | int | 10 | GSI平滑参数 |

## 性能提升

根据原始论文，升级后的StrongSORT++相比基础版本：
- **MOT17**: HOTA提升1.3-2.2%
- **MOT20**: HOTA提升1.3-2.2%
- **计算开销**: AFLink 591.9Hz, GSI 140.9Hz

## 注意事项

1. **AFLink模型**: 需要下载预训练的AFLink模型文件
2. **GPU内存**: 启用AFLink会增加GPU内存使用
3. **向后兼容**: 原有API完全兼容，不会影响现有代码
4. **错误处理**: 如果增强功能初始化失败，会自动回退到基础模式

## 文件结构

```
strongsort/
├── __init__.py
├── strong_sort.py          # 主类（已升级）
├── gsi.py                  # 新增：GSI模块
├── aflink/                 # 新增：AFLink模块
│   ├── __init__.py
│   ├── app_free_link.py
│   ├── model.py
│   ├── dataset.py
│   └── config.py
└── ...                     # 其他原有文件
```

## 示例代码

详细的使用示例请参考 `example_enhanced_usage.py` 文件。 