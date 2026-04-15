import torch
torch.cuda.empty_cache()      # 清空缓存
torch.cuda.synchronize()      # 等待 GPU 完成