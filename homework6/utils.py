import os
import random
import numpy as np
import torch

def setup_directories():
    """创建结果保存目录"""
    dirs = ['runs', 'outputs', 'plots']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def check_gpu_memory():
    """检查 GPU 显存并返回可用显存（GB）"""
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU 显存总量: {total_mem:.2f} GB")
        return total_mem

def check_file_exists(filepath, description):
    """检查文件是否存在，若存在则返回 True，否则打印提示"""
    if os.path.exists(filepath):
        print(f" {description} 已存在: {filepath}")
        return True
    else:
        print(f" {description} 不存在: {filepath}")
        return False