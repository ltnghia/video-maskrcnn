import os
import math
from maskrcnn_benchmark.utils.comm import get_world_size
import torch
import random
import numpy as np


def get_device(cfg=None, str_device='cpu'):
    if cfg:
        return torch.device(cfg.MODEL.DEVICE)
    else:
        return torch.device(str_device)


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # return get_world_size()


def get_batch_size_per_gpu(num_batches):
    num_gpu = get_num_gpus()
    batch_size_per_gpu = num_batches // num_gpu
    if batch_size_per_gpu == 0:
        batch_size_per_gpu = 1
    return batch_size_per_gpu


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if get_num_gpus() > 1:
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

