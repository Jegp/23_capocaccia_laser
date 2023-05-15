import torch
import numpy as np
import laser
import aestream
import sdl
import time
thresh = 0.999
nAccumulateFrames = 40
maskSize = 20
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
    #F = torch.zeros((640,480))
    #for i in range(nAccumulateFrames):#
      # f = camera.read()
      # F = F + f
    # idxX = slice(max([1,whereAmI[0]-maskSize]),min([640,whereAmI[0] + maskSize]))
    # idxY = slice(max([1,whereAmI[1]-maskSize]),min([480,whereAmI[1] + maskSize]))
    # F[idxX,idxY] = 0
      
    idx = np.argmax(f)
    print(f.max())
    a,b = np.unravel_index(idx,f.shape)
    xdist = whereAmI[0] -a 
    ydist = whereAmI[1] - b
    whereAmI[0] -=xdist
    whereAmI[1] -=ydist
    ### l.move(whereAmI[0],whereAmI[1])

