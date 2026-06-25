import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQN(nn.Module):
    """
    The Deep Q-Network Architecture.
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        # TODO 1: Define your Neural Network layers here.
        # Input size: state_dim. Output size: action_dim.
        
        pass

    def forward(self, x):
        """
        Calculates the forward pass of the network to output raw Q-values.
        """
        # TODO 2: Implement the forward pass.
        
        pass

class DQNAgent:
    """
    The Agent that interacts with the environment using the DQN.
    """
    def __init__(self, state_dim, action_dim, device):
        self.action_dim = action_dim
        self.device = device
        
        # Initialize the Policy Network and move it to the correct device
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        
    def select_action(self, state, epsilon):
        """
        Selects an action using an epsilon-greedy policy.
        """
        # TODO 3: Implement epsilon-greedy action selection.
        # With probability epsilon, choose a random action.
        # Otherwise, use policy_net to choose the action with the maximum Q-value.
        
        pass