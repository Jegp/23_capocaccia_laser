import torch
import numpy as np
import laser
import aestream
import sdl
import time

l = laser.Laser()
resolution = (640,480)
whereAmI = np.array([300,300])
l.on()
l.move(whereAmI[0],whereAmI[1])
window, pixels = sdl.create_sdl_surface(*resolution)

with aestream.USBInput((640,480)) as camera:  

  while True:
    time.sleep(0.04)
    f = camera.read()
    pixels[:] = sdl.events_to_bw(f)
    window.refresh()

    # idxX = slice(max([1,whereAmI[0]-maskSize]),min([640,whereAmI[0] + maskSize]))
    # idxY = slice(max([1,whereAmI[1]-maskSize]),min([480,whereAmI[1] + maskSize]))
    # F[idxX,idxY] = 0
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
    y -=ydist
    l.move(x,y)

