import torch
import torch.nn as nn
import torch.nn.utils as utils

class DRQNModel(nn.Module):

    def __init__(self, mode='Train'):
        super(DRQNModel, self).__init__()

        # 2 layer CNN 
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU())

        # Dense layer w/ dropout (CNN -> reshape -> rnn input)
        self.drop = nn.Dropout(0.3) if mode == 'Train' else None

        # 1 Layer LSTM
        self.rnn = nn.LSTM(32*9*9, 256, num_layers=1, batch_first=True)
        self.state = None

        # Dense linear output
        self.out = nn.Linear(256, 6)

    
    def forward(self, x):
        if x.dim() < 5:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        max_seq = x.shape[1]

        # Apply 2 layer CNN
        x = x.reshape(-1, 1, 84, 84)
        x = self.conv(x)

        # Reshape to dense and apploy dropout if needed
        x = x.reshape(batch_size, max_seq, 32*9*9)
        if self.drop is not None:
            x = self.drop(x)

        # Apply RNN
        if self.state is None:
            x, self.state = self.rnn(x)
        else:
            (h, c) = self.state
            x, self.state = self.rnn(x, (h.detach(), c.detach()))
        
        # Apply output layer
        x = self.out(x)
        return x

    def reset_state(self):
        self.state = None

