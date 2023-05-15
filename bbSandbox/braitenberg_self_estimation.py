import torch, laser, aestream, sdl, time, filter_conv
import numpy as np

l = laser.Laser()
resolution = (640,480)
whereAmI = np.array([300,300])
l.on()

l.move(whereAmI[0],whereAmI[1])
window, pixels = sdl.create_sdl_surface(*resolution)

with aestream.USBInput((640,480)) as camera:  

  while True:
    time.sleep(nAccumulateFrames*1e-3)
    f = camera.read()
    pixels[:] = sdl.events_to_bw(f)
    window.refresh()
    #F = torch.zeros((640,480))
    #for i in range(nAccumulateFrames):#
      # f = camera.read()
      # F = F + f
    
    print(torch.max(f))
      
    idx_target = np.argmax(f)
    x_target, y_target = np.unravel_index(idx_target,f.shape)
    
    #print(x_target, y_target)
    
    f[max([1,x_target-maskSize]):min([resolution[0],x_target + maskSize]),max([1,y_target-maskSize]):min([resolution[1],y_target + maskSize])] = 0
    
    idx_me = np.argmax(f)
    x_me, y_me = np.unravel_index(idx_me, f.shape)
    
    #print(x_me, y_me)
    
    xdist = x_me - x_target
    ydist = y_me - y_target
    whereAmI[0] -= xdist
    whereAmI[1] -= ydist
    l.move(whereAmI[0],whereAmI[1])

