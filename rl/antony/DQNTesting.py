#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:57:51 2023

@author: antony
"""

import time
from aestream import USBInput
import sdl
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
import numpy as np
import laser
import torchvision.transforms as transforms

from PIL import Image

#TO BUILD NETWORK AND OTHER FUNCTIONS: see https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
import math
import random
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',

                        ('state', 'action', 'next_state', 'reward'))
# Get number of actions from gym action space
n_actions = 4

# class DQN(nn.Module):

#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         # print("x shape = {}".format(x.shape))
#         x = x.to(device)
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        """
            DQN = 3 convolutional layers + 1 feed forward layer"""
        #The parameters can be changed to have those written down in the paper
        self.conv1 = nn.Conv2d(1,32,kernel_size=8,stride=4)
        convw = self.conv2d_size_out(w,kernels=8,stride=4)
        convh = self.conv2d_size_out(h,kernels=8,stride=4)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4, stride=2)
        convw = self.conv2d_size_out(convw,kernels=4,stride=2)
        convh = self.conv2d_size_out(convh,kernels=4,stride=2)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)
        convw = self.conv2d_size_out(convw,kernels=3,stride=1)
        convh = self.conv2d_size_out(convh,kernels=3,stride=1)
        linear_input_size = convw*convh*64
        self.lin1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512,outputs)
        
        #Calculation of the output of the conv2d layers: 
    def conv2d_size_out(self, size, kernels=5, stride=2):
        return (size - (kernels))//stride +1        
        
    def forward(self, x):
        #pushing x to the device
        x = x.to(device)
        #applying relu after each layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(F.relu(self.lin1(x.view(x.size()[0],-1)))) #we need to put x into its proper size afterwards

#FUNCTIONS AND HYPERPARAMETERS

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``Adam`` optimizer

REPLAY_SIZE = 10000
LENGTH_MIN = 0
LENGTH_MIN2 = 1000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0 #0.02
EPS_DECAY = 1 #10000 #10000
# TARGET_UPDATE = 20
LR = 1e-4 #1e-4
# once = False

resize = transforms.Compose([transforms.ToPILImage(),transforms.Resize((84,84), interpolation=Image.BICUBIC), transforms.ToTensor()])

# Get the number of state observations
n_observations = 84*84 #640*480 #len(state)

# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)

target_net = DQN(84, 84, n_actions).to(device)
model_save_name = 'mod04.ext' #mod04.ext
target_net.load_state_dict(torch.load("/opt/rl/antony/" + model_save_name))
target_net.eval()


steps_done = 0

sample = np.random.default_rng()
def select_action(state, it):
    global steps_done
    samp = random.random()
    if(it>LENGTH_MIN):
        eps_threshold = max(EPS_END,EPS_START - steps_done / EPS_DECAY)

        # print(eps_threshold)
        steps_done += 1
        if samp > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("act = {}".format(policy_net(state).max(2)[0].view(1, 1).shape))
                # target_net(non_final_next_states).max(2)[0].squeeze(1)
                # print(policy_net(state).max(1)[1].view(1, 1))
                return target_net(state).max(1)[1].view(1, 1)
        else:
            r_number = sample.integers(low=0, high=n_actions, size=1)[0]
            # print(eps_threshold)
            r_number = torch.tensor([r_number], device=device, dtype=torch.long).view(1,1)
            # print(r_number)
            return r_number
    else:
        r_number = sample.integers(low=0, high=n_actions, size=1)[0]
        r_number = torch.tensor([r_number], device=device, dtype=torch.long).view(1,1)
        # print(r_number)
        return r_number


episode_durations = []


def plot_durations(show_result=False):
    # plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    # if show_result:
    #     plt.title('Result')
    # else:
    #     plt.clf()
    #     plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('Duration')
    # plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        # print(means)
        # plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())
    
if torch.cuda.is_available():
    num_episodes = 150 #50 #30 # 50
else:
    num_episodes = 150 #50 #30 # 50 

resolution = (640, 480)
nlasers = 2
thresh_end = 200
keys = ["w", "a", "s", "d"]#, "e"]
startp = 0
def parse_char(laser, ch, state=None):
    mv = 50 #100 #100
    moves = {"w": (-mv, 0), "s": (mv, 0), "a": (0, -mv), "d": (0, mv)}#, "e": (0, 0)}

    if state is None:
        state = (startp,startp)
    # print(ch)
    if ch in moves.keys():
        state = (state[0] + moves[ch][0], state[1] + moves[ch][1])
        state = (min(2000, max(0, state[0])), min(2000, max(0, state[1])))
        laser.move(*state)
    elif ch == "q":
        laser.off()
        return None
    else:
        print("Unknown input", bytes(ch, "ascii"))

    return state

thresh = 0
window, pixels = sdl.create_sdl_surface(*resolution)
window2, pixels2 = sdl.create_sdl_surface(*resolution)
tensor = []
with USBInput(resolution) as stream:
    with laser.Laser() as l:
        it = 0
        l.off()
        l.move(startp, startp)
        for i_episode in range(num_episodes):
            # Initialize the environment and get it's state
            #print(state.shape)
            coord = (startp, startp)
            rew = 0
            sum_rew = 0
            avg_loss = 0
            for t in count():
                # if(t>=501):
                #     break
                it+=1
                # time.sleep(0.5)
                # tensor = []
                # for e in range(4):            
                tensor = stream.read()
                # tensor = torch.Tensor(tensor)
                # tensor = torch.mean(tensor)
                # print(type(tensor))
                # print(tensor.shape) 
                # print(state.shape)
                tensor_nonflat = tensor.to(device)
                tensor = torch.flatten(tensor).to(device)
                state_cpu = np.transpose(np.array(np.nonzero(tensor.cpu().numpy())))
                tensor_nonflat = tensor_nonflat / torch.max(tensor_nonflat)
                # tensor_ag = torch.flatten(resize(tensor_nonflat)).unsqueeze(0)
                tensor_ag = resize(tensor_nonflat).unsqueeze(0)
                action = select_action(tensor_ag, it)
                if(state_cpu.shape[0] != 0):
                    l.on()
                    # time.sleep(0.5)            
                    l.off()
                    # time.sleep(0.005)

                    
                    # print(torch.max(tensor))
                    coord = parse_char(l, keys[action], state=coord)
                    # Read a tensor (640, 480) tensor from the camera
                    tensor_new = stream.read() 
                    tensor_np = tensor_new.numpy()
                    new_tensor = np.zeros((resolution))
                    tensor_np[tensor_np < thresh] = 0
                    pixels[:] = sdl.events_to_bw(torch.from_numpy(tensor_np))
                    window.refresh()
                    # time.sleep(0.3)
                    events = np.transpose(np.array(np.nonzero(tensor_np)))
                    # print(events)
                    # print("nb ev = {}, max = {}".format(np.sum(tensor_np), np.max(tensor_np)))
                    # print(events.shape)
                    # FIRST STEP : obtaining the coordinates of the centers
                    if(np.array(events).shape[0] >= 2):
                        v_1 = []
                        v_2 = []
                        new_ev = []
                        clustering = DBSCAN(eps=15, min_samples=15).fit(events)
                        # to fix reward
                        for j, e in enumerate(clustering.labels_):
                            if(e==0):
                                v_1.append(events[j])
                                new_tensor[events[j][0],events[j][1]] = 1 
                            elif(e==1):
                                v_2.append(events[j])
                                new_tensor[events[j][0],events[j][1]] = 1
                            # if(e==-1):
                            #     new_tensor[events[j][0],events[j][1]] = 0
                        reward = 0
                        if(len(v_1)>0 and len(v_2)> 0):
                            # print("shape v1 = {} and shape v2 = {}".format(np.array(v_1).shape, np.array(v_2).shape))
                            # print("real number of clusters = {}".format(clustering.n_features_in_))
                            pixels2[:] = sdl.events_to_bw(torch.from_numpy(np.array(new_tensor)))
                            window2.refresh()
                            centers = []
                            centers.append(np.mean(np.array(v_1),axis = 0))
                            centers.append(np.mean(np.array(v_2), axis = 0))
                            # print("clusters = {}".format(centers))
                            # print(centers[0][0])
                            distance = np.sqrt( (centers[0][0]-centers[1][0])**2 + (centers[0][1]-centers[1][1])**2 )
                            # print("distance = {}".format(distance))
                            # reward = (100 - distance)/50
                        else:
                            continue
                            # distance = 300
                        reward = int((150 - distance)/10) 
                        # print("reward = {}".format(reward))
                        sum_rew+=reward
                        # reward = 0
                        # if(distance >= 0 and distance < 20):
                        #     reward = 1
                        #     print(reward)
                        # if(distance >= 20 and distance < 70):
                        #     reward = 0
                        # if(distance >= 70 and distance < 200):
                        #     reward = -1
                        # if(distance >= 200):
                        #     reward = -2
                            
                        # print(reward)
                        
                    # SECOND STEP : using built neural network and performing 3-factor learning rules (NEST simulator)
                    # window.refresh()
                    # time.sleep(0.01)
                        # tensor_new = tensor_new / torch.max(tensor_new)
                        # tensor = torch.flatten(resize(tensor_new)).to(device).unsqueeze(0)
                        tensor = resize(tensor_new).to(device).unsqueeze(0)
                        observation = tensor
                        terminated = False
                        # print(t)
                        # if(distance <= 60): #distance >= thresh_end or t >= 300 or distance <= 50):
                            # print("distance = {}".format(distance))
                            # terminated = True
                            # print(terminated)
                        reward = torch.tensor([reward], device=device)
                        done = terminated
                        rew += reward

                        if done:
                            episode_durations.append(t + 1)
                            # plot_durations()
                            break
            if(t!=0):
                print("episode = {0}, loss = {1}, eps_threshold = {2}, reward = {3}, it = {4}".format(i_episode, avg_loss, max(EPS_END,EPS_START - steps_done / EPS_DECAY), sum_rew/t, it))
        # l.off()
    

print('Complete')
#plot_durations(show_result=True)
# plt.ioff()
# plt.show()