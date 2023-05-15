import torch
import numpy as np
import laser
import aestream
import sdl
import time

l = laser.Laser()
resolution = (640,480)
l.on()
window, pixels = sdl.create_sdl_surface(*resolution)

with aestream.USBInput((640,480)) as camera:  

  while True:
    time.sleep(5)
    f = camera.read()
    pixels[:] = sdl.events_to_bw(f)
    window.refresh()
    # The steered laser blinks, so it's lower intensity than the
    # target laser. 
    idx = np.argmax(f)
    print(f.max())
    a,b = np.unravel_index(idx,f.shape)  ## location of the target
    idx = np.where(f>90)
    f[idx] = 0
    idx = np.argmax(f)
    print(f.max)
    x,y = np.unravel_index(idx,f.shape)  ## location of steered laser
    xdist = x -a 
    ydist = y -b
    x -=xdist
    y +=ydist
 #   l.move(x,y)

