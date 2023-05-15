import time
from aestream import USBInput
import sdl
import laser
import torch
from scipy import signal
import numpy as np

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d
# Define our camera resolution
resolution = (640, 480)
dvsres = (640, 480, 2)
avgresolution = (160, 120)

# Initialize our canvas
#window, pixels = sdl.create_sdl_surface(*resolution)
#window3, pixels3 = sdl.create_sdl_surface(*resolution)
#window2, pixels2 = sdl.create_sdl_surface(*avgresolution)

count = 0
score = 0

def self_estimation(coor_1, coor_2, previous_target_coor, previous_control_coor, previous_command, proba):

    proba_increment = .01
    
    #coor_1 is my target
    
    print(f'coor 1 target: {coor_1}')
    print(f'previous target:{previous_target_coor}')
    
    v_1_target = coor_1 - previous_target_coor
    v_1_control = coor_1 - previous_control_coor
    v_2_target = coor_2 - previous_target_coor
    v_2_control = coor_2 - previous_control_coor
    
    v = torch.vstack([v_1_control.T, v_2_control.T, v_1_target.T, v_2_target.T])
    sim = (torch.matmul(v, previous_command)/torch.norm(v, dim=1)).abs()
    
    print(f'similarity values: {sim}')
    
    if sim.argmax()==0:
        coor_target, coor_control = coor_1, coor_2
        proba[0] += proba_increment
    elif sim.argmax()==1:
        coor_target, coor_control = coor_2, coor_1
        proba[0] += proba_increment
    elif sim.argmax()==2:
        coor_target, coor_control = coor_1, coor_2
        proba[0] -= proba_increment
    else:
        coor_target, coor_control = coor_2, coor_1
        proba[0] -= proba_increment
    proba /= proba.sum()
        
    return coor_target, coor_control, proba

# Start streaming from a DVS camera on USB 2:2
with USBInput(dvsres, device="cuda") as stream:
    with laser.Laser() as l:
    
        p = 1
        l.move(2000,2000)
        lastx = 2000
        lasty = 2000
        lasterrory = 0
        lasterrorx = 0
        offtensor = torch.zeros(dvsres, device="cuda")
        offavgmap = torch.zeros(avgresolution, device="cuda")
        gautensor = torch.zeros(resolution, device="cuda")
        #gridx = torch.arange(160)
        #gridy = torch.arange(120)
        #gridx = gridx[None, :].float()
        #gridy = gridy[:, None].float()
        avgpool2d = torch.nn.AvgPool2d((8,8), stride=(4,4), padding=(2,2))
        avgpool2don = torch.nn.AvgPool2d((32,32), stride=(4,4), padding=(14,14))
        kernel_size = 24
        gaukernel = torch.Tensor(gkern(kernlen=kernel_size, std=12)).to("cuda")
        xlimit = resolution[0]-1-kernel_size//2
        ylimit = resolution[1]-1-kernel_size//2
        
        previous_target_coor = torch.Tensor([0,0])
        previous_control_coor = torch.Tensor([0,0])
        proba = torch.Tensor([.5,.5])
        
        #l.off()
        l.on()
        #ontensor = stream.read()
        while True:
            # Read a tensor (640, 480) tensor from the camera
            #tensor = stream.read()
            # 1tensor = torch.randn(640, 480).float() + 2
            stayflag = 0
            
            l.off()
            l.on()
            #time.sleep(0.01)
            tensor = stream.read()
                        
            #time.sleep(0.005)
            #offtensor = stream.read()
            #l.on()
            
            time.sleep(0.005)
            ontensor = stream.read()
            offtensor = ontensor
            #l.off()
            
            offtensor = offtensor[:,:,0]+offtensor[:,:,1]
            tensor4d = offtensor[None, None, :, :]
            avgmap = avgpool2d(tensor4d)
            offavgmap = avgmap[0,0,:,:]
            #print((avgmap==torch.max(avgmap)).nonzero())
            offmapmax = torch.max(offavgmap)
            maxix = (offavgmap==offmapmax).nonzero()
            offmaxix = maxix[0,:]
            #print(torch.sum(offtensor))
            #print(f"off max = {offmapmax}")
            #pixels[:] = sdl.events_to_bw(offtensor.cpu())
            #window.refresh()
            if offmapmax < 3:
                stayflag = 1
            else:
            #else:
            #    normavgmap = offavgmap/torch.sum(offavgmap)
            #    #print(normavgmap.size())
            #    #print(gridx.size())
            #    tempx = torch.mm(gridx, normavgmap)
            #    print(tempx)
            #    centroidx = torch.sum(tempx)
            #    centroidy = torch.sum(torch.mm(normavgmap,gridy))
            #    print(f"centroid = {centroidx}, {centroidy}")
                
            #print(f"offmaxix = {offmaxix}")
            #print(maxix.size())
            #pixels2[:] = sdl.events_to_bw(offavgmap)
            #window2.refresh()
            
                gautensor[:] = 0
                xtemp = offmaxix[0]*4
                ytemp = offmaxix[1]*4
                if offmaxix[0] > xlimit:
                    xtemp = xlimit
                if offmaxix[1] > ylimit:
                    ytemp = ylimit
                if offmaxix[0] < kernel_size//2:
                    xtemp = kernel_size//2
                if offmaxix[1] < kernel_size//2:
                    ytemp = kernel_size//2
                    
                gautensor[xtemp-kernel_size//2:xtemp+kernel_size//2,ytemp-kernel_size//2:ytemp+kernel_size//2] = gaukernel
                ontensor = ontensor[:,:,0]+ontensor[:,:,1]
                ontensor = ontensor-offmapmax*gautensor
                ontensor[ontensor<0] = 0    
                tensor4d = ontensor[None, None, :, :]
                avgmap = avgpool2d(tensor4d)
                onavgmap = avgmap[0,0,:,:]
                #if abs(lasterrorx) >= 10 and abs(lasterrory) >= 10:
                #    onavgmap = onavgmap-1*mapmax*gautensor
                #elif lasterrorx == 0 and lasterrory == 0:
                #    onavgmap = onavgmap-1*mapmax*gautensor
                #else:
                #    onavgmap = onavgmap-0.2*mapmax*gautensor
                
                #onavgmap = onavgmap-1*mapmax*gautensor
                
                #onavgmap = onavgmap - 5*offavgmap
                #print(onavgmap.size())
                #print((avgmap==torch.max(avgmap)).nonzero())
                onmapmax = torch.max(onavgmap)
                maxix = (onavgmap==onmapmax).nonzero()
                onmaxix = torch.mean(maxix.float(), 0)
                #onmaxix = maxix[0,:]
                if onmapmax < 0.5:
                    stayflag = 1
                #print(f"on max = {onmapmax}")
                #print(f"offmaxix = {offmaxix}")
                #print(f"onmaxix = {onmaxix}")
                #print(maxix.size())
                #print(onmaxix.size())
                #pixels3[:] = sdl.events_to_bw(ontensor.cpu())
                #window3.refresh()
                #onavgmap[onavgmap<0] = 0    
                #pixels2[:] = sdl.events_to_bw(onavgmap.cpu())
                #window2.refresh()
                errory = offmaxix[0] - onmaxix[0]
                errorx = offmaxix[1] - onmaxix[1]
                errthr = 8
                if torch.abs(errorx) <= errthr and torch.abs(errory) <= errthr:
                    #stayflag = 1
                    speed = 0.4
                else:
                    speed = 1
                
                if stayflag == 0:
                    scaling_mvmt = 2
                    
                    diferrory = errory - lasterrory
                    diferrorx = errorx - lasterrorx
                    kp = scaling_mvmt*speed
                    kd = 0*speed
                    nextx = lastx+kp*errorx+kd*diferrorx
                    nexty = lasty+kp*errory+kd*diferrory
                    nextx = nextx.int()
                    nexty = nexty.int()
                    
                    if nextx>4095:
                        nextx = 4095
                    if nexty>4095:
                        nexty = 4095
                    if nextx < 0:
                        nextx = 0
                    if nexty < 0:
                        nexty = 0
                    
                    coor_1 = torch.Tensor([offmaxix[0], offmaxix[1]])
                    coor_2 = torch.Tensor([onmaxix[0], onmaxix[1]])
                    
                    previous_command = torch.Tensor([lasterrory,lasterrorx])
                    
                    coor_target, coor_control, proba = self_estimation(coor_1, coor_2, previous_target_coor, previous_control_coor, previous_command, proba)
                    
                    if (coor_target[0]==offmaxix[0])&(coor_target[1]==offmaxix[1]):
                        score += 1
                    
                    if count%1000:
                        print(f'probability: {proba}')
                        print(f'score: {score/count}')
                    
                    l.move(nextx, nexty)
                    lastx=nextx
                    lasty=nexty
                    lasterrory = errory
                    lasterrorx = errorx
                    
                    count += 1
                    
                else:
                    errorx = 0
                    errory = 0
                    lasterrorx = 0
                    lasterrory = 0
                #print(errory)
                #print(errorx)
                previous_target_coor = coor_target
                previous_control_coor = coor_control