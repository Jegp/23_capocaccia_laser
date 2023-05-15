import time
from aestream import USBInput
import sdl
import laser
import norse.torch as norse
import torch
from scipy import signal
import numpy as np

import dsnt

# Define our camera resolution
resolution = (640, 480)
image_res = (320, 240)

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(*image_res)

# Create kernel
kernel_size = 15
gauss = signal.gaussian(kernel_size, std=7)
kernel = np.outer(gauss, gauss)
kernel = (kernel - 0.8)
kernel = kernel / kernel.max()
kernel = torch.tensor(kernel, device="cuda", dtype=torch.float)

# Setup network
model = torch.nn.Sequential(
    torch.nn.AvgPool2d(2),
    torch.nn.Conv2d(1, 1, kernel_size, bias=False, padding="same"),
    # torch.nn.Linear(160 * 120, 2),
    # norse.LICell()
)
model[1].weight = torch.nn.Parameter(kernel.view(1, 1, kernel_size, kernel_size))
model = model.cuda()


with torch.inference_mode():
    with laser.Laser() as l:
        l.blink(10)
        with USBInput(resolution) as stream:
            state = None
            dsnt_m_s = None
            dsnt_p_s = None
            while True:
                # Read a tensor (640, 480) tensor from the camera
                tensor = stream.read()
                tensor = tensor.cuda()

                out = model(tensor.view(1, 1, 640, 480))
                
                mirror_out = out.clone()
                mirror_out[mirror_out > 400] = 0
                mirror_out[mirror_out < 200] = 0
                mirror_out[mirror_out > 0] = 1
                dsnt_m_c, dsnt_p_s = dsnt_mirror(mirror_out, dsnt_p_s)

                print(dsnt_m_c)

                # Render pixels
                out[out < 2] = 0
                pixels[:] = out.squeeze().cpu()
                window.refresh()
                time.sleep(0.01)
