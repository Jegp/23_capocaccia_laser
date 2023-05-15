import torch
import numpy as np
import laser
import aestream
import sdl
import time

l = laser.Laser()
resolution = (640,480)
l.off()
window, pixels = sdl.create_sdl_surface(*resolution)

with aestream.USBInput((640,480)) as camera:  
  N = 4
  M = 25
  humanThreshold = 100
  maskSize = 40
  time.sleep(N)
  f = camera.read()
  while f.max > humanThreshold  ## human laser is present -- start the loop
    ### mask out the human laser
    idx = np.argmax(f)
    a,b = np.unravel_index(idx,f.shape)  ## location of human target
    
    time.sleep(N)
    f = camera.read()
    pixels[:] = sdl.events_to_bw(f)
    window.refresh()
    # The steered laser blinks, so it's lower intensity than the
    # target laser. 

    print(f.max())
    
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

