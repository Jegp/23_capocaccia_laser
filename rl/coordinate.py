import time
from aestream import USBInput
#import torch
#from scipy import signal
import sdl

# Define our camera resolution
resolution = (640, 480)
window, pixels = sdl.create_sdl_surface(*resolution)

# Define network
#window = signal.gaussian(10, std=7)
#kernel = np.outer(window, window)
#kernel = (kernel - 0.8)
#kernel = kernel / kernel.max()
#c = torch.nn.Conv2d(1, 1, 10)
#c.weight = torch.nn.Parameter(kernel.view(1, 1, 10, 10))

# Start streaming from a DVS camera on USB 2:2
with USBInput(resolution, device="cuda") as stream:
    while True:
        # Read a tensor (640, 480) tensor from the camera
        tensor = stream.read()
        #filtered = c(tensor.view(1, 1, 640, 480)

        # Render pixels
        pixels[:] = sdl.events_to_bw(tensor)
        window.refresh()
        time.sleep(0.01)
