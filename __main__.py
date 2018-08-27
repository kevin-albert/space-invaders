import time
from datetime import timedelta
from itertools import count
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from drqn_model import DRQNModel
from drqn_env import SpaceInvaders
from replay_mem import ReplayMem
import drqn_policy

# References
# http://cs229.stanford.edu/proj2016/report/ChenYingLaird-DeepQLearningWithRecurrentNeuralNetwords-report.pdf
# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
# https://arxiv.org/pdf/1609.05521.pdf
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Decay future prediction 
gamma       = 0.99

# Frame skip
k           = 3

# Training params
episodes    = 100_000       # total episodes to play
batch       = 32            # minibatch size
trace       = 20            # playback sequence depth - 1 second
load_every  = 5             # every n episodes, update target net
print_every = 1             # every n episodes, print out progress
save_every  = 10            # every n episodes, save checkpoint
init_steps  = 20            # Build up replay memory before training

# Initialize models
device      = torch.device('cpu')
policy_net  = DRQNModel(mode='Train').to(device)
target_net  = DRQNModel(mode='Eval').to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer   = optim.Adam(policy_net.parameters(), lr=1e-4, weight_decay=1e-6)

# Environment
env = SpaceInvaders(k)

# Replay memory
mem = ReplayMem()

# Reporting and such
Q_avg = np.zeros(episodes)
scores = np.zeros(episodes)
losses = np.zeros(episodes)
total_frames = 0

def init_replay_mem():
    seq = []

    for _ in range(init_steps):
        env.reset()
        for t in count():
            S = env.get_state().to(device)
            a = drqn_policy.random_action()
            r, done = env.step(a)
            
            a = torch.tensor(a, dtype=torch.int32).to(device)
            r = torch.tensor(r).to(device)
            seq.append((S, a, r))
            
            if done:
                (S, a, r) = map(torch.stack, zip(*seq))
                mem.store(S, a, r)
                break


def train_batch(episode):
    global total_frames

    policy_net.reset_state()
    target_net.reset_state()

    # Sample from replay memory
    (S, a, r, t) = mem.sample(batch, trace)
    
    # Evaluate training batch
    pred = policy_net(S[:, :-1]).gather(2, a[:, :-1])
    pred_target = target_net(S[:, 1:]).gather(2, a[:, 1:]).detach()

    # Only train on second half of replay trace
    depth = int(trace/2)
    pred = pred[:, -depth:].squeeze()
    pred_target = pred_target[:, -depth:]
    r = r[:, -depth:]

    # Compute r + (gamma)Q(s, a{t+1})
    not_t = 1 - t.type(torch.float32)

    target = r + gamma * pred_target[:, -1] * not_t.reshape(batch, 1)

    # Calculate loss and backpropagate
    loss = F.smooth_l1_loss(pred, target)
    losses[episode] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Running total of frames played
    total_frames += batch * trace * k


def play_episode(episode):
    global total_frames

    env.reset()
    target_net.reset_state()
    seq = []

    Q_sum = 0
    score = 0

    for t in count():
        S = env.get_state().to(device)
        p = target_net(S).squeeze()
        Q_sum += p.max(0)[0]
        Q_action = p.argmax().item()
        a = drqn_policy.epsilon_greedy(episode, Q_action)
        r, done = env.step(a)
        a = torch.tensor(a, dtype=torch.int32).to(device)
        r = torch.tensor(r).to(device)
        seq.append((S, a, r))
        score += r
        if done:
            (S, a, r) = map(torch.stack, zip(*seq))
            mem.store(S, a, r)
            break

    Q_avg[episode] = Q_sum / t
    scores[episode] = score
    total_frames += t * k


def train(episode):
    play_episode(episode)
    train_batch(episode)

    if episode % load_every == 0:
        # Copy params to target_next
        target_net.load_state_dict(policy_net.state_dict())

    if episode % print_every == 0:
        # Print current progress
        total_time = timedelta(seconds=int(total_frames / 60))
        print('{}| episode={} loss={:<3.5f} Q={:<3.5f} score={:<5}'.format(
            total_time, episode, losses[episode], Q_avg[episode], \
            scores[episode]))
        
    if episode % save_every == 0:
        # Save state
        save(episode, './drqn_checkpoint')


def save(episode, filename):
    torch.save({
        'episode': episode,
        'model': policy_net.state_dict(),
        'mem': mem.state_dict(),
    }, filename)

print('Initializing replay memory')
init_replay_mem()

print('Training')
for episode in range(episodes):
    train(episode)

print('Saving final model')
save(episodes, './drqn_final')

print('Done')