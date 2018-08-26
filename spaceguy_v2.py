import math
import random
import time
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import gym
import gym.spaces

device = torch.device('cuda')

# General hyperparameters
SEQ = 200
N_HIDDEN = 256
GAMMA = 0.96        # Long term vs short term
EPOCHS = 1000_000   # Epochs
DOWNSAMPLE = 3      # Framerate downsampling
SIZE = 84           # Image width x height

# Parameterize an exponential curve for epsilon
EPSILON_START = 0.9
EPSILON_END = 0.1
EPSILON_STEPS = 10_000
EPSILON_EXP_BASE = math.pow(EPSILON_END/EPSILON_START, 1/EPSILON_STEPS)

# Misc. constants
print_steps = 100
save_steps = 100

def epsilon(step):
    if step >= EPSILON_STEPS:
        return EPSILON_END
    return EPSILON_START * math.pow(EPSILON_EXP_BASE, step)

class RnnModel(nn.Module):

    def __init__(self, mode='Train'):
        super(RnnModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
        )
        p_drop = 0.3 if mode == 'Train' else 0
        self.drop = nn.Dropout(p_drop)
        self.rnn = nn.LSTM(32*9*9, N_HIDDEN, num_layers=1)
        self.fc = nn.Linear(N_HIDDEN, 6)
    
    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.conv(x)
        x = x.reshape(-1, 1, 32*9*9)
        x = self.drop(x)
        h, c = self.state
        x, self.state = self.rnn(x, (h.detach(), c.detach()))
        return F.relu(self.fc(x))

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

def save_checkpoint(experiment, filename):
    torch.save({
        'step': experiment.step,
        'episode': experiment.episode,
        'model': experiment.model.state_dict()
    }, filename)

def load_checkpoint(filename, mode='Train'):
    checkpoint = torch.load(filename)
    step = checkpoint['step']
    episode = checkpoint['episode']
    state_dict = checkpoint['model']
    model = RnnModel(mode)
    model.load_state_dict(state_dict)
    return RecurrentQNetwork(step, episode, model)


class RecurrentQNetwork:

    def __init__(self, step, episode, model):
        self.step = step
        self.episode = episode
        self.episode_frame = 0
        self.model = model.to(device)
        self.last_state = (torch.zeros(1, 1, N_HIDDEN).to(device),
                           torch.zeros(1, 1, N_HIDDEN).to(device))
        self.model.set_state(self.last_state)

        self.S = torch.zeros(SEQ, 1, 84, 84).to(device)
        self.y = torch.zeros(SEQ, 6).to(device)
        self.a = torch.zeros(SEQ, 1, dtype=torch.int64).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, 
                                    weight_decay=1e-6)
        self.Q_sum = 0
        self.r_sum = 0

        # Image stuff
        self.resize = T.Compose([T.ToPILImage(),
                                 T.Grayscale(),
                                 T.Resize(84, interpolation=Image.CUBIC)])

        # gym stuff
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()
        self.S = self.get_screen()

    def train_step(self):
        y_ = torch.zeros(SEQ, 6).to(device)      
        self.y = torch.zeros(SEQ, 6).to(device)
        seq = 0
        # Play through a batch of k-frames
        for i in range(SEQ):    
            seq += 1        
            # Take action with epsilon-greedy strategy
            y_[i] = self.model(self.S)
            if random.random() < epsilon(self.step):
                self.a[i] = action = random.randint(0, 5)
            else:
                self.a[i] = action = y_[i].argmax().item()
            
            reward = 0
            for j in range(DOWNSAMPLE):  
                _, r, done, _ = self.env.step(action)
                reward += r
                if done:
                    break
            self.y[i, action] = reward

            # Target includes prediction of future reward
            if i > 0 and not done:
                Q_future = y_[i].detach().max()
                self.y[i-1] += GAMMA * Q_future
            
            # For reporting
            self.r_sum += reward
            self.Q_sum += y_[i, action].item()
            self.episode_frame += 1
            if done:
                d = datetime.timedelta(seconds = self.step / 20.0 * SEQ)

                Q_avg = self.Q_sum / self.episode_frame
                print('{} game #{:<6} score={:<4} Q={:<3.5f} ϵ={}'.format(
                    d, self.episode, int(self.r_sum), Q_avg, epsilon(self.step)))
                self.r_sum = self.Q_sum = 0
                self.env.reset()
                self.model.set_state((torch.zeros(1, 1, N_HIDDEN).to(device),
                                      torch.zeros(1, 1, N_HIDDEN).to(device)))
                self.episode += 1
                self.episode_frame = 0
                break

        self.last_state = self.model.get_state()
        S_next = self.get_screen()

        if not done:
            Q_future = self.model(S_next).detach().max().squeeze()
            self.y[-1] += GAMMA * Q_future

        # optimize

        y_ = y_[0:seq].gather(1, self.a[0:seq])
        y = self.y[0:seq].gather(1, self.a[0:seq])
        loss = F.mse_loss(y_, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # prepare for next step
        self.S = S_next
        self.model.set_state(self.last_state)
        self.step += 1

    def get_screen(self):
        # transpose into torch order (CHW)
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1)) 
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        screen = self.resize(screen)
        screen = TF.crop(screen, 26, 0, SIZE, SIZE)
        screen = TF.to_tensor(screen)
        return screen.to(device)


    def free_play(self):
        self.env.reset()
        self.model.set_state((torch.zeros(1, 1, N_HIDDEN).to(device),
                              torch.zeros(1, 1, N_HIDDEN).to(device)))
        action = 0
        step_time = 1.0/60
        for i in range(100_000):
            t_next = time.time() + step_time;
            if i % DOWNSAMPLE == 0:
                self.env.render()
            
                # execute the model
                x = self.get_screen()
                p = self.model(x)
                action = p.argmax().item()
            _, _, done, _ = self.env.step(action)
            if done:
                self.env.reset()
                self.model.set_state(
                        (torch.zeros(1, 1, N_HIDDEN).to(device),
                         torch.zeros(1, 1, N_HIDDEN).to(device)))
                time.sleep(3)
            t_now = time.time()
            if t_next > t_now:
                time.sleep(t_next-t_now)
        

experiment = RecurrentQNetwork(0, 0, RnnModel())
#experiment = load_checkpoint('checkpoint_v2_200')
for i in range(1_000_000):
    experiment.train_step()
    if i % save_steps == 0:
        save_checkpoint(experiment, 'checkpoint_v2_200')

experiment = load_checkpoint('checkpoint_v2_200', mode='Test')
experiment.free_play()

