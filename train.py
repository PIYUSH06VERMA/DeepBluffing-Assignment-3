import gymnasium as gym
import math
import random
import numpy as np
import torch
import torch.optim as optim
import os

from src import DQNAgent, ReplayBuffer, optimize_model

# ---------------------------------------------------------
# TODO 0: Hyperparameter Tuning
# Tune these values to get your agent to solve the environment.
# Hint: CartPole-v1 is considered solved when the average reward 
# over 100 consecutive episodes is >= 195.0.
# ---------------------------------------------------------
BATCH_SIZE = None       # Typical values: 32, 64, 128, 256
GAMMA = None            # Discount factor (close to 1.0, e.g., 0.95 to 0.999)
EPS_START = None        # Initial exploration rate (usually 0.9 or 1.0)
EPS_END = None          # Minimum exploration rate (usually 0.01 to 0.05)
EPS_DECAY = None        # Controls speed of decay (higher means slower decay, try 500-2000)
TARGET_UPDATE = None    # How often to update target net in episodes (try 5-20)
LR = None               # Learning rate for AdamW (try 1e-4 to 1e-3)
MEMORY_SIZE = None      # Capacity of Replay Buffer (try 5000 to 20000)

MAX_EPISODES = 1000
SOLVED_THRESHOLD = 195.0 

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

def main():
    # Basic check to make sure they filled in the hyperparameters
    if any(v is None for v in [BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, LR, MEMORY_SIZE]):
        raise ValueError("Please fill in all hyperparameters in TODO 0 before running the script.")

    # Initialize Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize Agent and Networks
    agent = DQNAgent(state_dim, action_dim, device)
    target_net = agent.policy_net.__class__(state_dim, action_dim).to(device)
    target_net.load_state_dict(agent.policy_net.state_dict())
    target_net.eval()
    
    # Initialize Optimizer and Memory
    optimizer = optim.AdamW(agent.policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    steps_done = 0
    episode_rewards = []
    
    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Calculate Epsilon for Exploration vs. Exploitation
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            # ---------------------------------------------------------
            # TODO 1: Select an action using the agent and the current epsilon
            # ---------------------------------------------------------
            
            
            # ---------------------------------------------------------
            # TODO 2: Step the environment using the selected action
            # Hint: capture next_state, reward, done, truncated, and info
            # ---------------------------------------------------------
            
            
            # ---------------------------------------------------------
            # TODO 3: Push the transition into the replay memory
            # Hint: Cast 'done' to a float when storing it
            # ---------------------------------------------------------
            
            
            # Move to the next state
            # state = ...
            
            # ---------------------------------------------------------
            # TODO 4: Call optimize_model to train the policy network
            # ---------------------------------------------------------
            
            
        episode_rewards.append(total_reward)
        
        # ---------------------------------------------------------
        # TODO 5: Update the target network
        # Hint: If the current episode is a multiple of TARGET_UPDATE, 
        # copy the state_dict from the policy_net to the target_net.
        # ---------------------------------------------------------
        
            
        # Calculate trailing 100-episode average for tracking
        recent_avg = np.mean(episode_rewards[-100:])
        if episode % 10 == 0:
            print(f"Episode {episode}\tReward: {total_reward:.1f}\tAvg(100): {recent_avg:.1f}\tEpsilon: {epsilon:.2f}")
        
        # Check if environment is solved
        if len(episode_rewards) >= 100 and recent_avg >= SOLVED_THRESHOLD:
            print(f"\nEnvironment solved in {episode} episodes!")
            
            # Save the trained weights
            os.makedirs("weights", exist_ok=True)
            torch.save(agent.policy_net.state_dict(), "weights/dqn_cartpole.pth")
            print("Model weights saved to 'weights/dqn_cartpole.pth'")
            break

    env.close()

if __name__ == "__main__":
    main()