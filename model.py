import torch
import torch.nn as nn
import torch.nn.functional as F

HEIGHT = 84
WIDTH = 84
RNN_HIDDEN = 256

class Policy(nn.Module):

    def __init__(self, device):
        super(Policy, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
        )
        self.rnn = nn.LSTM(32*9*9 + 6, RNN_HIDDEN, num_layers=1, 
                           batch_first=True)
        self.fc = nn.Linear(RNN_HIDDEN, 6)
        self.reset(device)
    
    def forward(self, x, action):
        x = self.conv(x)
        x = x.reshape(1, -1, 32*9*9)
        action = action.unsqueeze(0)

        x = torch.cat([x, action], 2)
        x, self.h = self.rnn(x, (self.h[0].detach(), self.h[1].detach()))
        return F.relu(self.fc(x))

    def reset(self, device):
        self.h = (torch.zeros(1, 1, RNN_HIDDEN).to(device), 
                  torch.zeros(1, 1, RNN_HIDDEN).to(device))