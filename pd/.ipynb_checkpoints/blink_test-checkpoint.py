import time
from aestream import USBInput
import sdl
import laser

# Define our camera resolution
resolution = (640, 480)

# Initialize our canvas
window, pixels = sdl.create_sdl_surface(*resolution)

# Start streaming from a DVS camera on USB 2:2
with laser.Laser() as l:
    l.on()
    y_base = 1000
    x_base = 1000
    square_shift = 2000
    nb_repetitions = 10
    step = 100
    l.blink(1)
    #for i in range(nb_repetitions):
    
    while True:
        l.move(x_base,y_base)
        time.sleep(0.002)
        l.move(x_base+square_shift,y_base+square_shift)
        time.sleep(0.002)

