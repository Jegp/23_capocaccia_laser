import torch
import norse.torch as norse
import dsnt
from scipy import signal
import numpy as np

def create_kernel(kernel_size, *variances):
  kernels = []
  for var in variances:
    gauss = signal.gaussian(kernel_size, std=var)
    kernel = np.outer(gauss, gauss)
    kernel = kernel / kernel.max()
    kernel = (kernel - 0.5)
    kernels.append(torch.from_numpy(kernel).float())
  return torch.nn.Parameter(torch.stack(kernels).view(len(variances), 1, kernel_size, kernel_size))


class LaserFilter(torch.nn.Module):
  
  def __init__(self):
    super().__init__()
    # Setup network
    kernel_size = 15
    # t_kernel = 
    p = norse.LIParameters(tau_syn_inv=1000,tau_mem_inv=900)
    self.model = norse.SequentialState(
        torch.nn.AvgPool2d((4, 3)),
        norse.LICell(p),
        torch.nn.Conv2d(1, 2, kernel_size, bias=False, padding="same"),
        torch.nn.ReLU(),
    )
    self.model[-3].weight = create_kernel(kernel_size, 3)

  def forward(self, x, state = None):
    y, state = self.model(x, state)

    # Mirror
    mirror = y.clone()
    mirror[mirror > 5] = 0
    mirror[mirror < 0] = 0

    # Target
    target = y.clone()
    target[target < 10] = 0
    return torch.stack([y.squeeze(), y.squeeze()]), state
