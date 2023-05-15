import time
from aestream import USBInput
import sdl

# Define our camera resolution
resolution = (640, 480)
window, pixels = sdl.create_sdl_surface(*resolution)

# Start streaming from a DVS camera on USB 2:2
with USBInput(resolution) as stream:
    while True:
        tensor = stream.read()

        # Render pixels
        pixels[:] = sdl.events_to_bw(tensor)
        window.refresh()
        time.sleep(0.01)