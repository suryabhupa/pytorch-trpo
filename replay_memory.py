import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

FullTransition = namedtuple('FullTransition', ('state', 'action', 'mask', 'next_state', 'reward', 'rollout', 'action_prob'))

class FullMemory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(FullTransition(*args))

    def sample(self):
        return FullTransition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

RolloutTransition = namedtuple('RolloutTransition', ('state', 'action', 'mask', 'next_state', 'reward', 'action_prob'))

class RolloutMemory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(RolloutTransition(*args))

    def sample(self):
        return RolloutTransition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

