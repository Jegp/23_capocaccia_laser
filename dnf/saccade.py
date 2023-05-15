import time
from aestream import USBInput
import sdl
import laser
import norse.torch as norse
import torch
import numpy as np

import dnf
import dsnt
import filtering

if __name__ == "__main__":
  # Define our camera resolution
  resolution = (640, 480)

  # Initialize our canvas
  window, pixels = sdl.create_sdl_surface(*resolution)

  # Setup model
  model = norse.SequentialState(
      filtering.LaserFilter(),
      dnf.MotorWeights(160, 160)
  ).cuda()
  with torch.inference_mode():
    with laser.Laser() as l:
      x = 0
      y = 0
      l.move(x, y)
      l.blink(10)
      state = None
      with USBInput(resolution, device="cuda") as stream:    
          while True:
              # Read a tensor (640, 480) tensor from the camera
              tensor = stream.read()
              (dx, dy), state = model(tensor.view(1, 1, 640, 480), state)
              x += dx
              y += dy
              l.move(int(x), int(y))
              # Render pixels)
              out = tensor.detach().cpu()
              pixels[:] = sdl.events_to_bw(out)
              window.refresh()
              time.sleep(0.002)
