import torch
# Reference to https://saturncloud.io/blog/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel/
class CacheManager:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
