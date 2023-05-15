import time
from aestream import USBInput
import sdl
import laser
import torch

# Define our camera resolution
resolution = (640, 480)
dvsres = (640, 480, 2)
avgresolution = (160, 120)

def self_estimation(coor_1, coor_2, previous_target_coor, previous_control_coor, previous_command, p_control, p_target):

    proba_increment = .01
    
    v_1_target = coor_1 - previous_target_coor
    v_1_control = coor_1 - previous_control_coor
    v_2_target = coor_2 - previous_target_coor
    v_2_control = coor_2 - previous_control_coor
    
    v = torch.vstack([v_1_control.T, v_2_control.T, v_1_target.T, v_2_target.T])
    sim = torch.matmul(v, previous_command)/torch.norm(v, dim=1)
    
    if sim.argmax()==0:
        coor_target, coor_control = coor_1, coor_2
        p_control += proba_increment
        p_control /= p_control + p_target
        p_target /= p_control + p_target
    elif sim.argmax()==1:
        coor_target, coor_control = coor_2, coor_1
        p_control += proba_increment
        p_control /= p_control + p_target
        p_target /= p_control + p_target
    elif sim.argmax()==2:
        coor_target, coor_control = coor_1, coor_2
        p_control -= proba_increment
        p_control /= p_control + p_target
        p_target /= p_control + p_target
    else:
        coor_target, coor_control = coor_2, coor_1
        p_control -= proba_increment
        p_control /= p_control + p_target
        p_target /= p_control + p_target
        
    return coor_target, coor_control, p_target, p_control
    
    
# Start streaming from a DVS camera on USB 2:2
with USBInput(dvsres, device="cuda") as stream:
    with laser.Laser() as l:
    
        p = 1
        l.move(2000,2000)
        lastx = 2000
        lasty = 2000
        lasterrory = 0
        lasterrorx = 0
        offtensor = torch.zeros(dvsres)
        offavgmap = torch.zeros(avgresolution)
        #gridx = torch.arange(160)
        #gridy = torch.arange(120)
        #gridx = gridx[None, :].float()
        #gridy = gridy[:, None].float()
        avgpool2d = torch.nn.AvgPool2d((8,8), stride=(4,4), padding=(2,2))
        
        l.off()
        ontensor = stream.read()
        while True:
            # Read a tensor (640, 480) tensor from the camera
            #tensor = stream.read()
            # 1tensor = torch.randn(640, 480).float() + 2
            stayflag = 0
            
            
            #time.sleep(0.01)
            tensor = stream.read()
                        
            
            time.sleep(0.006)
            offtensor = stream.read()
            l.on()
            
            time.sleep(0.006)
            ontensor = stream.read()
            l.off()
            
            offtensor = offtensor[:,:,0]
            tensor4d = offtensor[None, None, :, :]
            avgmap = avgpool2d(tensor4d)
            offavgmap = avgmap[0,0,:,:]
            #print((avgmap==torch.max(avgmap)).nonzero())
            mapmax = torch.max(offavgmap)
            maxix = (offavgmap==mapmax).nonzero()
            offmaxix = maxix[0,:]
            #print(torch.sum(offtensor))
            #print(f"off max = {mapmax}")
            if mapmax < 1:
                stayflag = 1
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
            #pixels[:] = sdl.events_to_bw(offtensor.cpu())
            #print(pixels[300,200])
            #halfboxsize = 10
            #if maxix[0]>halfboxsize and maxix[1]<640-halfboxsize:
            #    for i in range(-halfboxsize,halfboxsize):
            #        for j in range(-halfboxsize,halfboxsize):
            #            pixels[(maxix[0]+halfboxsize),(maxix[1]+halfboxsize)] = 255
            #window.refresh()
            #pixels2[:] = sdl.events_to_bw(offavgmap)
            #window2.refresh()
            
            
            #tensor = tensor-offtensor
            ontensor = ontensor[:,:,0]
            tensor4d = ontensor[None, None, :, :]
            avgmap = avgpool2d(tensor4d)
            onavgmap = avgmap[0,0,:,:]
            #onavgmap = onavgmap
            onavgmap = onavgmap - 5*offavgmap
            #print(onavgmap.size())
            #print((avgmap==torch.max(avgmap)).nonzero())
            mapmax = torch.max(onavgmap)
            maxix = (onavgmap==mapmax).nonzero()
            onmaxix = maxix[0,:]
            if mapmax < 0.5:
                stayflag = 1
            #print(f"on max = {mapmax}")
            #print(f"onmaxix = {onmaxix}")
            #print(maxix.size())
            #pixels3[:] = sdl.events_to_bw(ontensor.cpu())
            #window3.refresh()
            #onavgmap[onavgmap<0] = 0    
            #pixels2[:] = sdl.events_to_bw(onavgmap.cpu())
            #window2.refresh()
            errory = offmaxix[0] - onmaxix[0]
            errorx = offmaxix[1] - onmaxix[1]
            errthr = 3
            if torch.abs(errorx) <= errthr and torch.abs(errory) <= errthr:
                stayflag = 1
            
            if stayflag == 0:
                diferrory = errory - lasterrory
                diferrorx = errorx - lasterrorx
                kp = 20
                kd = 4
                nextx = lastx+kp*errorx+kd*diferrorx
                nexty = lasty+kp*errory+kd*diferrory
                
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
                lasterrory = errory
                lasterrorx = errorx
                
                #print('motor')
                #print(lasty)
                #print(lastx)
            else:
                errorx = 0
                errory = 0
                lasterrorx = 0
                lasterrory = 0
            #print(errory)
            #print(errorx)
