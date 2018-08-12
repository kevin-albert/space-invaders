import torch
import torch.nn as nn
import torchvision

RNN_HIDDEN = 256

class Model(nn.Module):

    def __init__(self, batch):
        super(Model, self).__init__()
        self.conv = torchvision.models.squeezenet1_1(pretrained=True)
        self.rnn = nn.LSTM(1006, RNN_HIDDEN, num_layers=3, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, 6)
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