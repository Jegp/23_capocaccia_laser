import time
from aestream import USBInput
import sdl
import laser
import torch

# Define our camera resolution
resolution = (640, 480, 2)

# Start streaming from a DVS camera on USB 2:2
with USBInput(resolution, device="cuda") as stream:
    with laser.Laser() as l:
        while True:
            #time.sleep(0.005)
            tensor = stream.read()
            torch.cuda.synchronize()
            print(tensor.sum())
            print(tensor.size())