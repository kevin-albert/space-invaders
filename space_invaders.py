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


# init game environment
env = gym.make('SpaceInvaders-v0')

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
BATCH = 1
SEQ = 100
CHANNELS = 3
HEIGHT = 224
WIDTH = 224
RNN_HIDDEN = 256
OUTPUTS = env.action_space.n


class Model(nn.Module):

    def __init__(self, batch):
        super(Model, self).__init__()
        self.conv = torchvision.models.squeezenet1_1(pretrained=True)
        self.rnn = nn.LSTM(1006, RNN_HIDDEN, num_layers=3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, env.action_space.n)
        self.batch = batch
        self.reset()
    
    def forward(self, x, action):
        batch = x.size(0)
        seq = x.size(1)
        x = x.view(-1, CHANNELS, HEIGHT, WIDTH)
        x = self.conv(x).detach()  # just use pretrained embedding
        x = x.view(batch, seq, -1)
        x = torch.cat([x, action], 2)
        x, self.h = self.rnn(x, (self.h[0].detach(), self.h[1].detach()))
        return F.relu(self.fc(x))

    def reset(self):
        self.h = (torch.zeros(3, self.batch, RNN_HIDDEN), 
                  torch.zeros(3, self.batch, RNN_HIDDEN))

model_train = Model(BATCH)
model_eval = Model(1)
model_eval.load_state_dict(model_train.state_dict())

resize = T.Compose([T.ToPILImage(),
                    T.Resize((HEIGHT, WIDTH), interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen).unsqueeze(0)
    return screen


def get_action(step, screen, action):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * step / EPS_DECAY)
    sample = random.random()
    action_t = torch.zeros(1, 1, 6)
    action_t[0, 0, action] = 1
    action_next = model_eval(screen.unsqueeze(0), action_t)
    if sample > eps_threshold:
        return torch.argmax(action_next.squeeze()).item()
    else:
        return env.action_space.sample()


def get_seq(step):
    x = torch.zeros(BATCH, SEQ, 3, 224, 224)
    y = torch.zeros(BATCH, SEQ, 6) 
    actions = torch.zeros(BATCH, SEQ, 6)
    env.reset()
    action = 0
    for i in range(BATCH):
        for j in range(SEQ):
            env.render()
            scr = get_screen()
            x[i, j] = scr
            action = get_action(step, scr, action)
            print('action', action)
            _, reward, done, _ = env.step(action)
            y[i, j, action] = reward
            actions[i, j, action] = 1
            if done:
                env.reset()
                break
        
    return x, y, actions

epochs = 1000
print_every = 1
optimizer = optim.Adam(model_train.parameters())
env.reset()

for epoch in range(epochs):
    x, y, actions = get_seq(epoch)
    # model.reset()
    y_ = model_train(x, actions).view(-1, OUTPUTS)

    actions_id = torch.argmax(actions, 2).view(-1, 1)
    y = y.view(-1, 6)
    y_ = y_.view(-1, 6)

    loss = F.smooth_l1_loss(y_.gather(1, actions_id), y.gather(1, actions_id))

    if epoch % print_every == 0:
        print('{:>5}: {}'.format(epoch, loss.item()))
        model_eval.load_state_dict(model_train.state_dict())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()