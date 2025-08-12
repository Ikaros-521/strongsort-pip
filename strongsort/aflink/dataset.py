"""
@Author: Du Yunhao
@Filename: dataset.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Dataset for AFLink
"""
import os
import torch
import numpy as np
from os.path import join, exists
from collections import defaultdict
from sklearn.preprocessing import normalize


class LinkData:
    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.data = self.load_data()
    
    def load_data(self):
        """加载数据"""
        data = []
        if exists(self.root):
            for file in os.listdir(self.root):
                if file.endswith('.txt'):
                    file_path = join(self.root, file)
                    tracks = np.loadtxt(file_path, delimiter=',')
                    data.append(tracks)
        return data
    
    def transform(self, track1, track2):
        """
        数据预处理和特征提取
        
        Args:
            track1: 轨迹1数据
            track2: 轨迹2数据
        
        Returns:
            feat1: 轨迹1特征
            feat2: 轨迹2特征
        """
        # 提取轨迹特征
        feat1 = self.extract_track_features(track1)
        feat2 = self.extract_track_features(track2)
        
        # 转换为tensor
        feat1 = torch.FloatTensor(feat1)
        feat2 = torch.FloatTensor(feat2)
        
        return feat1, feat2
    
    def extract_track_features(self, track):
        """
        提取轨迹特征
        
        Args:
            track: 轨迹数据 (N, 5) [frame, x, y, w, h]
        
        Returns:
            features: 特征向量 (10,)
        """
        if len(track) == 0:
            return np.zeros(10)
        
        # 计算轨迹统计特征
        frames = track[:, 0]
        positions = track[:, 1:3]  # x, y
        sizes = track[:, 3:5]      # w, h
        
        # 位置特征
        pos_mean = np.mean(positions, axis=0)
        pos_std = np.std(positions, axis=0)
        pos_range = np.max(positions, axis=0) - np.min(positions, axis=0)
        
        # 尺寸特征
        size_mean = np.mean(sizes, axis=0)
        size_std = np.std(sizes, axis=0)
        
        # 时间特征
        duration = frames[-1] - frames[0] + 1
        speed = np.linalg.norm(pos_range) / max(duration, 1)
        
        # 组合特征
        features = np.concatenate([
            pos_mean,      # 2: 平均位置
            pos_std,       # 2: 位置标准差
            pos_range,     # 2: 位置范围
            size_mean,     # 2: 平均尺寸
            size_std,      # 2: 尺寸标准差
            [duration],    # 1: 持续时间
            [speed]        # 1: 平均速度
        ])
        
        return features 