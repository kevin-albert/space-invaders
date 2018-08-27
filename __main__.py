import time
from datetime import timedelta
from itertools import count
import os
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
load_every  = 1             # every n episodes, update target net
print_every = 1             # every n episodes, print out progress
save_every  = 5             # every n episodes, save checkpoint
init_steps  = 20            # Build up replay memory before training

# Initialize models
if torch.cuda.is_available():
    device  = torch.device('cuda')
    print('CUDA detected - using GPU')
else:
    device  = torch.device('cpu')
    print('CUDA not detected - using CPU')

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
    policy_net.reset_state()
    target_net.reset_state()

    # Sample from replay memory
    (S, a, r, t) = mem.sample(batch, trace, device)
    
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


def play_episode(episode):
    global total_frames

    env.reset()
    target_net.reset_state()
    seq = []

    Q_sum = 0
    score = 0
    p_sum = torch.zeros(6).to(device)

    for t in count():
        S = env.get_state().to(device)
        p = target_net(S).squeeze()
        p_sum += p.detach()
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

    p_sum /= t
    # sanity check
    #print('p avg: [ ', end='')
    #for i in range(6):
    #    print('{:2.5f} '.format(p_sum[i].item()), end='')
    #print(']', end='')
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
        print('| {} | episode={} loss={:<3.5f} Q={:<3.5f} score={:<5}'.format(
            total_time, episode, losses[episode], Q_avg[episode], \
            scores[episode]))
        
    if episode % save_every == 0:
        # Save state
        save(episode, './checkpoints/spaceguy')

def save(episode, filename):
    global total_frames
    torch.save({
        'episode': episode,
        'model': policy_net.state_dict(),
        'mem': mem.state_dict(),
        'total_frames': total_frames
    }, filename)

def load(filename):
    global total_frames

    state = torch.load(filename)
    policy_net.load_state_dict(state['model'])
    target_net.load_state_dict(state['model'])
    mem.load_state_dict(state['mem'])
    if 'total_frames' in state:
        total_frames = state['total_frames']
    else:
        total_frames = 0

    return state['episode']

def free_play(games=1):
    env.reset()
    target_net.reset_state()
    t_frame = 1/60
    a = 0
    score = 0
    games_played = 0
    for t in count():
        t0 = time.time()
        env.render()
        S = env.get_state()
        if t % k == 0:
            # Execute the policy and take a new action
            p = target_net(S).squeeze()
            a = p.argmax().item()
        r, done = env.step(a)
        score += r
        if done:
            print('score = {}'.format(score))
            score = 0
            env.reset()
            target_net.reset_state()
            time.sleep(1)
            games_played += 1
            if games_played >= games:
                return
        else:
            t_diff = t_frame - (time.time() - t0)
            if t_diff > 0:
                time.sleep(t_diff)


if os.path.exists('./checkpoints/spaceguy'):
    print('Loading from checkpoint')
    ep_start = load('./checkpoints/spaceguy')
else:
    print('Initializing replay memory')
    init_replay_mem()
    ep_start = 0

print('Replay memory size: {}MB'.format(mem.size/1024/1024))
print('Training')

for episode in range(ep_start, episodes):
    train(episode)

print('Saving final model')
save(episodes, './checkpoints/spaceguy_final')

print('Free play...')
free_play(10)
env.close()

print('Done')
