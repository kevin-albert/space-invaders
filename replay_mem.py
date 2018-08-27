import torch
import random

# Storing 1x84x84 images plus actions and rewards should take ip just under
# 2GB or GPU ram.
N = 1_000

class ReplayMem:

    def __init__(self):
        self.data = []
        self.i = -0
        self.maxlen = 0

    def __len__(self):
        return len(self.data)

    def state_dict(self):
        return { 'data': self.data, 'i': self.i, 'maxlen': self.maxlen }

    def load_state_dict(self, state_dict):
        self.data = state_dict['data']
        self.i = state_dict['i']
        self.maxlen = state_dict['maxlen']

    def store(self, S, a, r):
        """
        Store a sequence of states, actions, and rewards.
        params:
        `S`: tensor of shape (seq, 1, 84, 84)
        `a`: tensor of shape (seq) for actions
        `r`: tensor of shape (seq) for rewards
        """
        if len(self.data) < N:
            self.i = len(self.data)
            self.data.append((S, a, r))
        else:
            self.i += 1
            if self.i >= N:
                self.i = 0
            self.data[self.i] = (S, a, r)
        if S.shape[0] > self.maxlen:
            self.maxlen = S.shape[0]
    
    def sample(self, D, L):
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
        if L > self.maxlen:
            raise RuntimeError('No sequence of length {} available'.format(L))

        S = torch.zeros(D, L, 1, 84, 84)
        a = torch.zeros(D, L, 1, dtype=torch.int64)
        r = torch.zeros(D, L)
        t = torch.zeros(D, dtype=torch.uint8)

        for i in range(D):
            while True:
                (Si, ai, ri) = random.choice(self.data)
                l = Si.shape[0]
                if l < L:
                    continue
                j = random.randint(0, l-L-1)
                k = j+L

                S[i] = Si[j:k]
                a[i] = ai[j:k].reshape(L, 1)
                r[i] = ri[j:k]

                # are we at the end of the sequence?
                t[i] = 1 if k == l else 0 
                break
        return S, a, r, t

    

