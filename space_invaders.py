import gym
import gym.spaces

import numpy as np
import math
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import Policy

import matplotlib.pyplot as plt

# init game environment
env = gym.make('SpaceInvaders-v0')

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
SEQ = 20
K = 3
HEIGHT = 84
WIDTH = 84
OUTPUTS = env.action_space.n

device = torch.device('cuda')

policy_train = Policy(device).to(device)
policy_target = Policy(device).to(device)
policy_target.load_state_dict(policy_train.state_dict())

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize(84, interpolation=Image.CUBIC)])

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen)
    screen = TF.crop(screen, 26, 0, HEIGHT, WIDTH)
    screen = TF.to_tensor(screen)
    return screen.to(device)


def get_action(step, screen, action):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * step / EPS_DECAY)
    sample = random.random()
    action_t = torch.zeros(1, 6).to(device)
    action_t[0, action] = 1
    action_next = policy_target(screen.unsqueeze(0), action_t)
    if sample > eps_threshold:
        return torch.argmax(action_next.squeeze()).item()
    else:
        return env.action_space.sample()

x = torch.zeros(SEQ+1, 1, 84, 84).to(device)
y = torch.zeros(SEQ+1, 6).to(device)
a = torch.zeros(SEQ+1, 6).to(device)
t = torch.zeros(SEQ+1).to(device)
total_reward = 0

def get_seq(step, should_print):
    global x, y, a, t, total_reward
    x[0] = x[-1]; x[1:] = 0
    y[0] = y[-1]; y[1:] = 0
    a[0] = a[-1]; a[1:] = 0
    t[0] = t[-1]; t[1:] = 0
    action = 0
    for i in range(1, SEQ+1):
        if should_print:
            env.render()
        scr = get_screen()
        x[i] = scr
        action = get_action(step, scr, action)
        reward = 0
        done = False
        for _ in range(K):
            _, reward_k, done, _ = env.step(action)
            reward += reward_k
            if done:
                break
        
        total_reward += reward
        if done:
            print('game over', total_reward)
            total_reward = 0
        
        y[i, action] = reward
        a[i, action] = 1
        if done:
            t[i] = 1
            env.reset()
            break
        
    return x, y, a, t

epochs = 1_000_000
print_every = 10
optimizer = optim.Adam(policy_train.parameters())
env.reset()

for epoch in range(epochs):
    should_print = epoch % print_every == 0

    x, y, a, t = get_seq(epoch, should_print)
    print('y[0]', y[0], 'a[0]', a[0])
    print('y[n]', y[-1], 'a[n]', a[-1])
    y_ = policy_train(x, a).view(-1, OUTPUTS)

    a_id = torch.argmax(a, 1).view(-1, 1).narrow(0, 0, SEQ)
    y = y.view(-1, 6)
    y_ = y_.view(-1, 6)


    # Get the indices of non-terminal rewards & predictions
    not_t = (torch.ones(SEQ).to(device) - t.narrow(0, 1, SEQ)).reshape(-1, 1)
    y_after = y_.narrow(0, 1, SEQ)
    y_0 = y_.narrow(0, 0, SEQ)
    y0 = y.narrow(0, 1, SEQ)
    y0.add_(y_after.detach() * not_t.expand_as(y_after) * 0.9)

    loss = F.smooth_l1_loss(y_0.gather(1, a_id), y0.gather(1, a_id))

    if should_print:
        print('{:>8}: {}'.format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policy_target.load_state_dict(policy_train.state_dict())
