"""print cuda:0 when PyTorch can use an NVIDIA GPU, else cpu (used by the launcher)"""
try:
    import torch
    print("cuda:0" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
