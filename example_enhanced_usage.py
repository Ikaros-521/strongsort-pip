"""
升级版StrongSORT使用示例
包含GSI和AFLink功能
"""

import cv2
import numpy as np
from strongsort import StrongSORT

def main():
    # 基础配置
    model_weights = 'osnet_x0_25_msmt17.pt'  # ReID模型权重
    device = 'cuda'  # 或 'cpu'
    fp16 = False
    
    # 创建升级版StrongSORT追踪器
    tracker = StrongSORT(
        model_weights=model_weights,
        device=device,
        fp16=fp16,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
        # 新增功能配置
        enable_aflink=True,           # 启用AFLink
        enable_gsi=True,              # 启用GSI
        aflink_model_path='AFLink_epoch20.pth',  # AFLink模型路径
        gsi_interval=20,              # GSI插值间隔
        gsi_tau=10,                   # GSI平滑参数
    )
    
    # 模拟检测结果
    # 格式: [x1, y1, x2, y2, confidence, class_id]
    detections = np.array([
        [100, 100, 200, 200, 0.9, 0],
        [300, 300, 400, 400, 0.8, 0],
        [500, 500, 600, 600, 0.7, 0],
    ])
    
    # 模拟图像
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # 更新追踪器
    tracks = tracker.update(detections, image)
    
    print("追踪结果:")
    print("格式: [x1, y1, x2, y2, track_id, class_id, confidence]")
    for track in tracks:
        print(f"  {track}")
    
    print("\n功能说明:")
    print("1. 基础StrongSORT: 多目标追踪")
    print("2. GSI (Gaussian-smoothed interpolation): 高斯平滑插值，补偿缺失检测")
    print("3. AFLink (Appearance-free link): 无外观信息的轨迹关联")

def example_without_enhancements():
    """不使用增强功能的示例"""
    tracker = StrongSORT(
        model_weights='osnet_x0_25_msmt17.pt',
        device='cuda',
        fp16=False,
        enable_aflink=False,  # 禁用AFLink
        enable_gsi=False,     # 禁用GSI
    )
    print("使用基础StrongSORT功能")

def example_gsi_only():
    """仅使用GSI功能的示例"""
    tracker = StrongSORT(
        model_weights='osnet_x0_25_msmt17.pt',
        device='cuda',
        fp16=False,
        enable_aflink=False,  # 禁用AFLink
        enable_gsi=True,      # 启用GSI
        gsi_interval=20,
        gsi_tau=10,
    )
    print("使用StrongSORT + GSI功能")

def example_aflink_only():
    """仅使用AFLink功能的示例"""
    tracker = StrongSORT(
        model_weights='osnet_x0_25_msmt17.pt',
        device='cuda',
        fp16=False,
        enable_aflink=True,           # 启用AFLink
        enable_gsi=False,             # 禁用GSI
        aflink_model_path='AFLink_epoch20.pth',
    )
    print("使用StrongSORT + AFLink功能")

if __name__ == "__main__":
    print("=== 升级版StrongSORT使用示例 ===\n")
    
    print("1. 完整功能示例:")
    main()
    
    print("\n2. 基础功能示例:")
    example_without_enhancements()
    
    print("\n3. 仅GSI功能示例:")
    example_gsi_only()
    
    print("\n4. 仅AFLink功能示例:")
    example_aflink_only() 