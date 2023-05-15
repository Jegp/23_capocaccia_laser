import torch
from filter_conv import LaserFilter
import numpy as np
import laser
import aestream
import sdl
import time

l = laser.Laser()
resolution = (640,480)
l.off()
window, pixels = sdl.create_sdl_surface(*resolution)
window2, pixels2 = sdl.create_sdl_surface(*resolution)

kernel_size = 15

avgpool2d = torch.nn.AvgPool2d((kernel_size,kernel_size))#, stride=(4,4), padding=(2,2))

with aestream.USBInput((640,480)) as camera:  
    N = 0.002  ### units of seconds
    M = 0.025  ### 25 msec

    while True:  
        time.sleep(N)
        f = camera.read()
        pixels[:] = sdl.events_to_bw(f)
        window.refresh()
        
        print(f)
        filtering = avgpool2d(kernel_size)
        
        print(filtering.forward(f))
        
        
   
        

  
