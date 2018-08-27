import torch
import random

# 1 GB replay memory
capacity = 1 * 1024 * 1024 * 1024

class ReplayMem:

    def __init__(self):
        self.data = []
        self.size = 0

    def __len__(self):
        return len(self.data)

    def state_dict(self):
        return { 'data': self.data }

    def load_state_dict(self, state_dict):
        self.data = []
        self.size = 0
        for (S, a, r) in state_dict['data']:
            self.store(S, a, r)

    def store(self, S, a, r):
        """
        Store a sequence of states, actions, and rewards.
        params:
        `S`: tensor of shape (seq, 1, 84, 84)
        `a`: tensor of shape (seq) for actions
        `r`: tensor of shape (seq) for rewards
        """
        
        self.data.append((S, a, r))
        self.size += self.tuple_size(S, a, r)
        self.cleanup()
            
    def cleanup(self):
        while self.size > capacity:
            (S, a, r) = self.data[0]
            self.data = self.data[1:]
            self.size -= self.tuple_size(S, a, r)

    def tuple_size(self, S, a, r):
        return 4 * (S.numel() + a.numel() + r.numel())
    
    def sample(self, D, L, device):
        """
        Sample batch of (state, action, reward, done) sequences
        params:
        `D`: batch size
        `L`: Sequence length

        returns:
        `S`: tensor of shape (D, L, 1, 84, 84)
        `a`: tensor of shape (D, L, 1), type int64
        `r`: tensor of shape (D, L)
        `t`: tensor of shape (D), type uint8
        """

        S = torch.zeros(D, L, 1, 84, 84).to(device)
        a = torch.zeros(D, L, 1, dtype=torch.int64).to(device)
        r = torch.zeros(D, L).to(device)
        t = torch.zeros(D, dtype=torch.uint8).to(device)

        for i in range(D):
            (Si, ai, ri) = random.choice(self.data)
            l = Si.shape[0]
            
            j = random.randint(0, l-L-1)
            k = j+L

            S[i] = Si[j:k]
            a[i] = ai[j:k].reshape(L, 1)
            r[i] = ri[j:k]

            # are we at the end of the sequence?
            t[i] = 1 if k == l else 0 
            break
        return S, a, r, t

    

