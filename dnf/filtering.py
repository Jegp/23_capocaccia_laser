import torch
import norse.torch as norse
import dsnt
from scipy import signal
import numpy as np
import math

def create_kernel(kernel_size, *variances):
  kernels = []
  for var in variances:
    gauss = signal.gaussian(kernel_size, std=var)
    kernel = np.outer(gauss, gauss)
    kernel = kernel / kernel.max()
    kernel = (kernel - 0.5)
    kernels.append(torch.from_numpy(kernel).float().cuda())
  return torch.nn.Parameter(torch.stack(kernels).view(len(variances), 1, kernel_size, kernel_size))

class ThresholdFilter(torch.nn.Module):

  def __init__(self, mirror_th = 3, target_th = 6):
    super().__init__()
    self.mirror_th = mirror_th
    self.target_th = target_th

  def forward(self, x):
    # Target
    target = x.clone()
    target[target < self.target_th] = 0

    # Mirror
    mirror = x.clone()
    mirror[mirror > self.mirror_th] = 0
    mirror[mirror < 1] = 0
    mirror = torch.relu(mirror - target)

    return torch.stack([target, mirror])

class LaserFilter(torch.nn.Module):
  
  def __init__(self):
    super().__init__()
    # Setup network
    kernel_size = 9
    p = norse.LIParameters(tau_syn_inv=1000,tau_mem_inv=800)
    self.model = norse.SequentialState(
        torch.nn.AvgPool2d((4, 3)),
        norse.LICell(p),
        torch.nn.Conv2d(1, 1, kernel_size, bias=False, padding="same"),
        torch.nn.ReLU(),
        ThresholdFilter()
    )
    self.model[-3].weight = create_kernel(kernel_size, 3)

  def coordinates(self, x):
    idx = x.flatten().argmax()
    return torch.tensor([idx // 160, idx % 160])

  def forward(self, x, state = None):
    (it, im), state = self.model(x, state)
    ct = self.coordinates(it)
    cm = self.coordinates(im)
    diff = cm - ct
    print(diff)
    co_image = torch.zeros_like(it).squeeze()
    coo_x = min(159, diff[0] + 80)
    coo_y = min(159, diff[1] + 80)
    co_image[coo_x, coo_y] = 1

    return co_image, state
