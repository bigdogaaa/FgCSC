import torch
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)         # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)             # 为numpy设置随机种子
