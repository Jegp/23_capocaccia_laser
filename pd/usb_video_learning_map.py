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
dvsres = (640, 480)
avgresolution = (160, 120)
debug = 0

count = 0
predicted = 0

# Initialize our canvas
if debug == 1:
    window, pixels = sdl.create_sdl_surface(*resolution)
    window3, pixels3 = sdl.create_sdl_surface(*resolution)
    window2, pixels2 = sdl.create_sdl_surface(*avgresolution)
    
M_target, M_source = torch.randn(2,2).to('cuda'), torch.randn(2,2).to('cuda')
previous_target, previous_control, previous_command = torch.ones(2), torch.ones(2), torch.zeros(2)
    
stored_laser, stored_point = torch.Tensor([2000,2000]).to('cuda'), torch.Tensor([2000,2000]).to('cuda')
    
def learning_map(M_source, X_1, previous_control, delta_laser):
    
    learning_rate = .000001
    
    #command = torch.hstack([previous_command[1], previous_command[0], torch.ones(1)/2])
    
    delta_dvs = X_1-previous_control

    delta_estimate = torch.matmul(M_source, delta_laser)

    gradient_a = -2*delta_laser[0]*(delta_dvs[0]-delta_estimate[0])
    gradient_b = -2*delta_laser[1]*(delta_dvs[0]-delta_estimate[0])
    gradient_c = -2*delta_laser[0]*(delta_dvs[1]-delta_estimate[1])
    gradient_d = -2*delta_laser[1]*(delta_dvs[1]-delta_estimate[1])
    
    #gradient_x = -2*previous_command*estimate[0]
    #gradient_y = -2*previous_command*estimate[1]

    # sure it can be done in one line ...
    M_source[0,0] -= learning_rate*gradient_a
    M_source[0,1] -= learning_rate*gradient_b 
    M_source[1,0] -= learning_rate*gradient_c
    M_source[1,1] -= learning_rate*gradient_d
    
    loss = torch.linalg.norm(delta_dvs-delta_estimate)
    
    #print(f'loss: {loss}')
    #print(f"M={M_source}")
    #print(f'estimate: {delta_estimate+previous_control}')
    #print(f'measure: {X_1}')
    #print(f'deltas: dvs {delta_dvs} - laser {delta_laser}')
    
    return M_source, loss
    
#def learning_map(M_target, M_source, X_1, X_2, previous_target, previous_source, previous_command):#, loss_1, loss_2):
    
    # take minimum Vs
    
    # learning phase 
    # compute loss_1 loss_2
    # choose the source as the point that minimizes the error
    

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
        avgloss = 1000
        avglossratio = 0.1
        confidence = 0
        
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
            
            #offtensor = offtensor[:,:,0]+offtensor[:,:,1]
            tensor4d = offtensor[None, None, :, :]
            avgmap = avgpool2d(tensor4d)
            offavgmap = avgmap[0,0,:,:]
            #print((avgmap==torch.max(avgmap)).nonzero())
            offmapmax = torch.max(offavgmap)
            maxix = (offavgmap==offmapmax).nonzero()
            offmaxix = maxix[0,:]
            #print(torch.sum(offtensor))
            if debug == 1:
                print(f"off max = {offmapmax}")
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
                if xtemp > xlimit:
                    xtemp = xlimit
                if ytemp > ylimit:
                    ytemp = ylimit
                if xtemp < kernel_size//2:
                    xtemp = kernel_size//2
                if ytemp < kernel_size//2:
                    ytemp = kernel_size//2
                    
                gautensor[xtemp-kernel_size//2:xtemp+kernel_size//2,ytemp-kernel_size//2:ytemp+kernel_size//2] = gaukernel
                #ontensor = ontensor[:,:,0]+ontensor[:,:,1]
                ontensor = ontensor-1*offmapmax*gautensor
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
                if debug == 1:
                    print(f"on max = {onmapmax}")
                    print(f"offmaxix = {offmaxix}")
                    print(f"onmaxix = {onmaxix}")
                    print(gaukernel[0,:])
                #print(maxix.size())
                #print(onmaxix.size())
                if debug == 1:
                    pixels3[:] = sdl.events_to_bw(ontensor.cpu())
                    window3.refresh()
                    #onavgmap[onavgmap<0] = 0    
                    pixels2[:] = sdl.events_to_bw(onavgmap.cpu())
                    window2.refresh()
                errory = offmaxix[0] - onmaxix[0]
                errorx = offmaxix[1] - onmaxix[1]
                errthr = 8
                if torch.abs(errorx) <= errthr and torch.abs(errory) <= errthr:
                    #stayflag = 1
                    speed = 0.4
                else:
                    speed = 1
                
                if stayflag == 0:
                    diferrory = errory - lasterrory
                    diferrorx = errorx - lasterrorx
                    kp = 2*speed
                    kd = 0*speed
                    nextx = lastx+kp*errorx+kd*diferrorx
                    nexty = lasty+kp*errory+kd*diferrory
                    nextx = nextx.int()
                    nexty = nexty.int()
                    
                    coor_target = torch.Tensor([offmaxix[0], offmaxix[1]])
                    coor_control = torch.Tensor([onmaxix[0], onmaxix[1]])
                    
                    #previous_command = torch.Tensor([nextx-lastx,nexty-lasty])
                    
                    #print(, coor_control, coor_target)
                    
                    #M_source, loss = learning_map(M_source, coor_control, previous_control, previous_command)
                    
                    #predicted = torch.matmul(M_source,torch.hstack([torch.Tensor([nextx,nexty]), torch.ones(1)/2]))
                    
                    
                    previous_control = coor_control
                    
                    #if (coor_target[0]==offmaxix[0])&(coor_target[1]==offmaxix[1]):
                    #    score += 1
                    
                    #if count<500 or (count%10==0 and count > 500):
                    if (count%10==0 and confidence==0) or (count%300==0 and confidence==1):
                        previous_command = torch.Tensor([lastx,lasty]).to('cuda')-stored_laser
                        M_source, loss = learning_map(M_source, onmaxix, stored_point, previous_command)
                        stored_laser = torch.Tensor([lastx,lasty]).to('cuda')
                        stored_point = onmaxix
                        avgloss = (1-avglossratio)*avgloss + loss*avglossratio
                        if avgloss < 4:
                            confidence = 1
                        else:
                            confidence = 0
                        print(f'avgloss: {avgloss}')
                        #print(f"target={coor_target}")
                        #print(f"count={count}")
                    count += 1
                        
                    #print(f"confidence = {confidence}")
                    #decision = torch.matmul(torch.linalg.inv(M_source).to('cuda'), torch.Tensor([errory,errorx]).to('cuda'))
                    decision_M = confidence*torch.matmul(torch.linalg.inv(M_source).to('cuda'), torch.Tensor([errory,errorx]).to('cuda'))
                    #decision_rand = (1-confidence)*400*(torch.rand(2)-torch.Tensor([0.5,0.5])).to('cuda')
                    decision_rand = torch.Tensor([kp*errorx, kp*errory]).to('cuda')
                    
                    decision = torch.Tensor([lastx, lasty]).to('cuda')+0.3*decision_M+decision_rand
                    nextx, nexty = decision[0].int(), decision[1].int()
                    
                    
                    if nextx>3000:
                        nextx = 3000
                    if nexty>4095:
                        nexty = 4095
                    if nextx < 0:
                        nextx = 0
                    if nexty < 0:
                        nexty = 0
                    
                    #if confidence == 1:
                        #print(f"inv1={torch.matmul(torch.linalg.inv(M_source), M_source)}")
                        #print(f"inv2={torch.matmul(M_source, torch.linalg.inv(M_source))}")
                        #print(f"error={[errory,errorx]}")
                        #print(f"decision_M={decision_M}")
                        #print(f"decision_rand={decision_rand}")
                        #print(f"nextx={nextx}, nexty={nexty}")
                    
                        
                    #decision = torch.matmul(torch.linalg.inv(M_source).to('cuda'), torch.Tensor([errory,errorx]).to('cuda'))
                    
                    #print(f'decision: {decision} - error: {errory,errorx}')
                    
                    l.move(nextx,nexty)
                    lastx=nextx
                    lasty=nexty
                    lasterrory = errory
                    lasterrorx = errorx

                else:
                    errorx = 0
                    errory = 0
                    lasterrorx = 0
                    lasterrory = 0
                    
                if debug == 1:
                    print(f"errorx = {errory}")
                    print(f"errory = {errorx}")
