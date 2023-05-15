import torch
import numpy as np
import laser
import aestream
import sdl
import time
import math

l = laser.Laser()
resolution = (640,480)
xNow = 2000
yNow = 2000
l.on()
l.move(xNow,yNow)
l.off()
#xdist = 0
#ydist = 0
window, pixels = sdl.create_sdl_surface(*resolution)
window2, pixels2 = sdl.create_sdl_surface(*resolution)

with aestream.USBInput((640,480)) as camera:  
  N = 0.005  ### units of seconds
  M = 0.025  ### 25 msec
  humanThreshold = 10 
  maskSize = 40

  ## find the hot pixels
  time.sleep(1)
  f = camera.read()
  flat_f = f.flatten()
  top_percentile = np.percentile(flat_f,99)
 ### top_values = flat_f[flat_f>top_percentile]
  maskHotPixels = np.where(f > top_percentile)
  
  while True:
  
   time.sleep(N)
   f = camera.read()
   f[maskHotPixels] = 0
   l.on()
   print(torch.max(f))
  
#   pixels[:] = sdl.events_to_bw(f)
#   window.refresh()
   
   if torch.max(f) > humanThreshold:  ## human laser is present -- start the loop
#    time.sleep(N)
#    f = camera.read()
#    f[maskHotPixels] = 0
#    print(torch.max(f))
#    pixels[:] = sdl.events_to_bw(f)
#    window.refresh()
   
    ### mask out the human laser
    idx = np.argmax(f)
    a,b = np.unravel_index(idx,f.shape)  ## location of human target
    print(f"a={a}, b={b}")
    time.sleep(N)
    f = camera.read()
    f[maskHotPixels]=0
    idxX = slice(max([1,a-maskSize]),min([640,a + maskSize]))
    idxY = slice(max([1,b-maskSize]),min([480,b + maskSize]))
 #   if math.sqrt(xdist**2 + ydist**2) >40:
    f[idxX,idxY] = 0
    idx = np.argmax(f)
    x,y = np.unravel_index(idx,f.shape)  ## location of steered laser
    print(f"x={x},y={y}")
 #   pixels2[:] = sdl.events_to_bw(f)
  #  window2.refresh()
    xdist = x-a 
    ydist = y-b
    sfX = 3
    sfY = 3
    xNow -=sfX*xdist
    yNow -=sfY*ydist
    if xNow > 4095:
      xNow = 4095
    if xNow < 0:
      xNow = 0
    if yNow > 4095:
      yNow = 4095
    if yNow < 0:
      yNow = 0
    l.move(yNow,xNow)
    time.sleep(M)
    l.off()
    time.sleep(0.05)
    f = camera.read()
#    time.sleep(0.002)
#    f = camera.read()
   # print('~~~~~~~~~~~~~~~~')
    #print(f.max())

  
