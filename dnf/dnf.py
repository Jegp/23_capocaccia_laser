import torch
import norse.torch as norse
from scipy import signal
import numpy as np

class SustainedField(torch.nn.Module):

  def __init__(self, kernel_size=11):
    super().__init__()
    self.kernel_size = kernel_size 
  
    gauss = signal.gaussian(self.kernel_size, std=7)
    kernel = np.outer(gauss, gauss)
    kernel = (kernel - 0.5)
    kernel = kernel / kernel.max()
    kernel = torch.tensor(kernel, device="cuda", dtype=torch.float)
    self.conv = torch.nn.Conv2d(1, 1, self.kernel_size, bias=False, padding="same")
    self.conv.weight = torch.nn.Parameter(kernel.view(1, 1, self.kernel_size, self.kernel_size))
    self.neuron = norse.LICell()

  def forward(self, input_tensor, state):
    input_tensor =  input_tensor[0]
    if state is None:
      x = input_tensor
      neuron_state = None
    else:
      x = (input_tensor + state[0] / state[0]).sigmoid() # Normalize
      neuron_state = state[1]
    y, neuron_state = self.neuron(self.conv(input_tensor), neuron_state)
    print(y.min(), y.max())
    return y, [y, neuron_state]

class MotorWeights(torch.nn.Module):

  def __init__(self, width, height):
    super().__init__()
    self.lin = torch.nn.Linear(width * height, 2, bias=False)
    weights = torch.linspace(5, 0, width // 2).unsqueeze(1).repeat(1, width)
    src = torch.concat([weights, weights.flip(0)])
    xw = src.flatten()
    yw = src.rot90().flatten()
    self.lin.weight = torch.nn.Parameter(torch.stack([xw, yw]))

  def forward(self, x):
    return self.lin(x.flatten())