import torch
from scipy import signal
import numpy as np

def create_kernel(kernel_size, variance=1):
    return torch.nn.Parameter(torch.Tensor([signal.gaussian(kernel_size, std=variance)]))

class LaserFilter(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # Setup network
        self.lpf = torch.nn.Conv2d(1, 1, kernel_size, bias=False, padding="same")
        self.lpf.weight = create_kernel(kernel_size)

    def forward(self, f):
        self.lpf(f)
        
