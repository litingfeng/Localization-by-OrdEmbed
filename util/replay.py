"""
Created on 3/31/2021 4:50 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
from collections import namedtuple
import random
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # def push(self, *args):
    #     """Saves a transition."""
    #     if len(self.memory) < self.capacity:
    #         self.memory.append(None)
    #     self.memory[self.position] = Transition(*args)
    #     self.position = (self.position + 1) % self.capacity
    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def push_batch(self, *args):
        #states, actions, next_states, rewards, dones = args
        sample_size = args[0].shape[0]
        if len(self.memory)+sample_size > self.capacity:
            del self.memory[:(len(self.memory)+sample_size - self.capacity)]
        self.memory += [Transition(state, action, next_state, reward, done) for
                        state, action, next_state, reward, done in
                        zip(*args)]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    import torch
    memory = ReplayMemory(3)
    state = torch.rand(2, 4)
    actions = torch.randint(0, 10, (2,))
    next_state = torch.rand(2, 4)
    rewards = torch.randint(0,2, (2,))
    done = torch.randint(0,2,(2,))
    print('state ', state)
    print('actions ', actions)
    print('next_state ', next_state)
    print('rewards ', rewards)
    print('done ', done)

    memory.push_batch(state, actions, next_state, rewards, done)
    print(memory.memory)
    state = torch.rand(2, 4)
    actions = torch.randint(0, 10, (2,))
    next_state = torch.rand(2, 4)
    rewards = torch.randint(0, 2, (2,))
    done = torch.randint(0, 2, (2,))
    print('2nd batch')
    print('state ', state)
    print('actions ', actions)
    print('next_state ', next_state)
    print('rewards ', rewards)
    print('done ', done)

    memory.push_batch(state, actions, next_state, rewards, done)
    print(memory.memory)

