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
  N = 1
  M = 25
  humanThreshold = 1000
  maskSize = 40

  ## find the hot pixels
  time.sleep(10)
  f = camera.read()
  flat_f = f.flatten()
  top_percentile = np.percentile(flat_f,99)
  top_values = flat_f[flat_f>top_percentile]
  maskHotPixels = np.where(f > top_percentile)
  
  while True:
  
   time.sleep(N)
   f = camera.read()
   f[maskHotPixels] = 0
   print(torch.max(f))
   pixels[:] = sdl.events_to_bw(f)
   window.refresh()

   if torch.max(f) > humanThreshold:
     print("f was greater than threshold")
     l.off()
   elif torch.max(f) < humanThreshold:
     print("f was smaller than threshold")
     l.on()
   
   """ while torch.max(f) > humanThreshold:  ## human laser is present -- start the loop
     time.sleep(0.5)
     print("I made it to the loop!")
     f = camera.read()
     print(torch.max(f))
     pixels[:] = sdl.events_to_bw(f)
     window.refresh()
   
    ### mask out the human laser
    idx = np.argmax(f)
    a,b = np.unravel_index(idx,f.shape)  ## location of human target
    l.on()
    time.sleep(N)
    f = camera.read()
    idxX = slice(max([1,a-maskSize]),min([640,a + maskSize]))
    idxY = slice(max([1,a-maskSize]),min([480,a + maskSize]))
    f[idxX,idxY] = 0
    idx = np.argmax(f)
    x,y = np.unravel_index(idx,f.shape)  ## location of human target
    pixels[:] = sdl.events_to_bw(f)
    window.refresh()
    xdist = x -a 
    ydist = y -b
    x -=xdist
    y +=ydist
    l.move(x,y)
    time.sleep(M)
    l.off()
    time.sleep(M)
    f = camera.read()
    time.sleep(1)
    f = camera.read()
    print('~~~~~~~~~~~~~~~~')
    print(f.max) """

  
