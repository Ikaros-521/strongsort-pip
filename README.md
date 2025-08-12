<div align="center">
<h1>
  StrongSort-Pip: Packaged version of StrongSort 
</h1>
<h4>
    <img width="700" alt="teaser" src="docs/uav.gif">
</h4>
<div>
    <a href="https://pepy.tech/project/strongsort"><img src="https://pepy.tech/badge/strongsort" alt="downloads"></a>
    <a href="https://badge.fury.io/py/strongsort"><img src="https://badge.fury.io/py/strongsort.svg" alt="pypi version"></a>
</div>
</div>

## <div align="center">Overview</div>

This repo is a packaged version of the [StrongSort](https://github.com/dyhBUPT/StrongSORT) algorithm with enhanced features including GSI (Gaussian-smoothed interpolation) and AFLink (Appearance-free link) for improved tracking performance.
### Installation
```
pip install strongsort
```

### Basic Usage
```python
from strongsort import StrongSORT

tracker = StrongSORT(model_weights='model.pth', device='cuda')
pred = model(img)
for i, det in enumerate(pred):
    det[i] = tracker[i].update(detection, im0s)
```

### Enhanced Usage with GSI and AFLink
```python
from strongsort import StrongSORT

# 创建升级版StrongSORT追踪器
tracker = StrongSORT(
    model_weights='osnet_x0_25_market1501.pth',
    device='cuda',
    enable_aflink=True,           # 启用AFLink
    enable_gsi=True,              # 启用GSI
    aflink_model_path='AFLink_epoch20.pth',  # AFLink模型路径
    gsi_interval=20,              # GSI插值间隔
    gsi_tau=10,                   # GSI平滑参数
)

# 更新追踪器
tracks = tracker.update(detections, image)
```

### Features
- **StrongSORT**: Base multi-object tracking algorithm
- **GSI (Gaussian-smoothed interpolation)**: Compensates for missing detections using Gaussian process regression
- **AFLink (Appearance-free link)**: Associates short tracklets into complete trajectories without appearance information

## Citations
```bibtex
@article{du2022strongsort,
  title={Strongsort: Make deepsort great again},
  author={Du, Yunhao and Song, Yang and Yang, Bo and Zhao, Yanyun},
  journal={arXiv preprint arXiv:2202.13514},
  year={2022}
}
```
