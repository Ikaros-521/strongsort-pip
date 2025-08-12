"""
@Author: Du Yunhao
@Filename: model.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Post Link Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import cfg


class PostLinker(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, output_dim=2):
        super(PostLinker, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 分类网络
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, track1, track2):
        """
        Args:
            track1: (batch_size, input_dim)
            track2: (batch_size, input_dim)
        Returns:
            logits: (batch_size, output_dim)
        """
        # 特征提取
        feat1 = self.feature_net(track1)
        feat2 = self.feature_net(track2)
        
        # 特征拼接
        feat_concat = torch.cat([feat1, feat2], dim=1)
        
        # 分类
        logits = self.classifier(feat_concat)
        
        return logits 