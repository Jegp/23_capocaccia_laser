import laser
import aestream
import time
import datetime
import numpy as np 

#l = laser.Laser()
cam_x, cam_y = 640, 480

x_coor = np.arange(0,cam_x,1)
y_coor = np.arange(0,cam_y,1)

with aestream.USBInput((cam_x,cam_y)) as camera:
    # In this case, we read() every 100ms
    interval = 0.5
    t_0 = time.time()
    
    #x_grid, y_grid = torch.meshgrid(torch.arange(0,cam_x,1), torch.arange(0,cam_y,1))
    #print(x_grid)

    # Loop forever
    while True:
        # When 500 ms passed...
        if t_0 + interval <= time.time():

            # Grab a tensor of the events arriving during the past 500ms
            frame = camera.read()

            # Reset the time so we're again counting to 500ms
            t_0 = time.time()

            # Sum the incoming events and print along the timestamp
            time_string = datetime.datetime.fromtimestamp(t_0).time()
            
            #x_weighted_coord = x_grid*frame
            #y_weighted_coord = y_grid*frame
            
            x_proj = frame.sum(axis=0)
            y_proj = frame.sum(axis=1)
            
            print(f"Frame at {time_string} with laser coordinates: {x_proj.argmax()} - {y_proj.argmax()}")
