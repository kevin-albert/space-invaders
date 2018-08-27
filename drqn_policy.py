import random

# Epsilon decay
E_start     = 1.0
E_end       = 0.1
E_steps     = 2000

def random_action():
    return random.randint(0, 5)

def epsilon_greedy(step, Q_action):
    if step > E_steps:
        e = E_start * (1.0 - step / E_steps) + E_end * (step / E_steps)
    else:
        e = E_end
    return Q_action if random.random() > e else random_action()
    