"""
AFLink configuration
"""
from easydict import EasyDict

# 创建配置对象
cfg = EasyDict()

# 模型保存目录
cfg.model_savedir = './checkpoints'

# 训练数据根目录
cfg.root_train = './data/train'

# 模型参数
cfg.input_dim = 10
cfg.hidden_dim = 256
cfg.output_dim = 2 