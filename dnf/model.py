import torch
import norse.torch as norse

import dnf

class SaccadeModel(torch.nn.Module):

  def __init__(self, width, height):
    super().__init__()
    self.perception = dnf.SustainedField(kernel_size=15)
    self.target = dnf.SustainedField(kernel_size=15)
    self.weights = dnf.CoordinateWeights(width, height)

  def forward(self, x):
    x, s = self.target(x)
    return x

  