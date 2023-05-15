import time
from aestream import USBInput
import sdl
import laser
import torch

# Define our camera resolution
resolution = (640, 480)
avgresolution = (160, 120)

# Initialize our canvas
#window, pixels = sdl.create_sdl_surface(*resolution)
#window2, pixels2 = sdl.create_sdl_surface(*avgresolution)

# Start streaming from a DVS camera on USB 2:2
with USBInput(resolution, device="gpu") as stream:
    with laser.Laser() as l:
    
        p = 1
        l.move(2000,2000)
        lastx = 2000
        lasty = 2000
        offtensor = torch.zeros(640, 480)
        offavgmap = torch.zeros(160, 120)
        gridx = torch.arange(160)
        gridy = torch.arange(120)
        avgpool2d = torch.nn.AvgPool2d((8,8), stride=(4,4), padding=(2,2))
        
        l.on()
        ontensor = stream.read()
        while True:
            # Read a tensor (640, 480) tensor from the camera
            #tensor = stream.read()
            # 1tensor = torch.randn(640, 480).float() + 2
            stayflag = 0
            
            
            time.sleep(0.01)
            tensor = stream.read()
                        
            
            time.sleep(0.005)
            offtensor = stream.read()
            
            l.on()
            l.off()
            time.sleep(0.005)
            ontensor = stream.read()
            
            tensor4d = offtensor[None, None, :, :]
            avgmap = avgpool2d(tensor4d)
            offavgmap = avgmap[0,0,:,:]
            #print((avgmap==torch.max(avgmap)).nonzero())
            mapmax = torch.max(offavgmap)
            maxix = (offavgmap==mapmax).nonzero()
            offmaxix = maxix[0,:]
            #print(f"off max = {mapmax}")
            if mapmax < 1:
                stayflag = 1
            #print(f"off max ix = {offmaxix}")
            #print(maxix.size())
            #pixels[:] = sdl.events_to_bw(offtensor)
            #print(pixels[300,200])
            #halfboxsize = 10
            #if maxix[0]>halfboxsize and maxix[1]<640-halfboxsize:
            #    for i in range(-halfboxsize,halfboxsize):
            #        for j in range(-halfboxsize,halfboxsize):
            #            pixels[(maxix[0]+halfboxsize),(maxix[1]+halfboxsize)] = 255
            #window.refresh()
            
            #tensor = tensor-offtensor
            tensor4d = ontensor[None, None, :, :]
            avgmap = avgpool2d(tensor4d)
            onavgmap = avgmap[0,0,:,:]
            onavgmap = onavgmap - 5*offavgmap
            #print(onavgmap.size())
            #print((avgmap==torch.max(avgmap)).nonzero())
            mapmax = torch.max(onavgmap)
            maxix = (onavgmap==mapmax).nonzero()
            onmaxix = maxix[0,:]
            if mapmax < 0.2:
                stayflag = 1
            #print(mapmax)
            #print(onmaxix)
            #print(maxix.size())
            #pixels[:] = sdl.events_to_bw(ontensor)
            #window.refresh()
            onavgmap[onavgmap<0] = 0    
            #pixels2[:] = sdl.events_to_bw(onavgmap)
            #window2.refresh()
            
            if stayflag == 0:
                lasterrory = errory
                lasterrorx = errorx
                errory = offmaxix[0] - onmaxix[0]
                errorx = offmaxix[1] - onmaxix[1]
                diferrory = errory - lasterrory
                diferrorx = errorx - lasterrorx
                kp = 20
                kd = 5
                nextx = lastx+kp*errorx+kd*diferrorx
                nexty = lasty+kp*errory+kd*diferrorx
                
                if nextx>4095:
                    nextx = 4095
                if nexty>4095:
                    nexty = 4095
                if nextx < 0:
                    nextx = 0
                if nexty < 0:
                    nexty = 0
                l.move(nextx, nexty)
                lastx=nextx
                lasty=nexty
                
                #print('motor')
                #print(lasty)
                #print(lastx)
            else:
                errorx = 0
                errory = 0
            #print(errory)
            #print(errorx)
