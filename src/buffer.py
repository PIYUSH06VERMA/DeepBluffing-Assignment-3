import random
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer to store and randomly sample transitions.
    """
    def __init__(self, capacity):
        # We use a deque with a fixed max length to automatically discard oldest memories
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Saves a transition step to memory.
        """
        # TODO 1: Append the transition tuple to self.memory
        pass

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions.
        """
        # TODO 2: Return a random sample of 'batch_size' transitions from self.memory
        pass

    def __len__(self):
        return len(self.memory)