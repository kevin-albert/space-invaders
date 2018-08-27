import gym
import gym.spaces

import numpy as np

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize(84, interpolation=Image.CUBIC)])

class SpaceInvaders:

    def __init__(self, k):
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()
        self.score = self.r = 0
        self.done = False
        self.k = k

    def reset(self):
        self.env.reset()
    
    def get_state(self):
        # transpose into torch order (CHW)
        S = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        S = np.ascontiguousarray(S, dtype=np.float32) / 255
        S = torch.from_numpy(S)
        S = resize(S)
        S = TF.crop(S, 26, 0, 84, 84)
        S = TF.to_tensor(S)
        return S

    def step(self, a):
        if self.done:
            self.score = self.r = 0
        self.r = 0
        for i in range(self.k):
            _, r, self.done, _ = self.env.step(a)
            self.r += r
            self.score += r
            if self.done:
                break
        return self.r, self.done

    def render(self):
        self.env.render()
        
    def action_space(self):
        return 6

    def close(self):
        self.env.close()