import gymnasium as gym
import torch
import numpy as np
import os
import time

from src import DQNAgent

# ---------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------
NUM_EVAL_EPISODES = 10
WEIGHTS_PATH = "weights/dqn_cartpole.pth"

# Setup device (Evaluation is lightweight, CPU is perfectly fine)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Verify that the weights file actually exists
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Could not find trained weights at '{WEIGHTS_PATH}'.")
        print("Please run 'python train.py' successfully to train and save the model first.")
        return

    # 2. Initialize the human-rendered environment for visual inspection
    print("\nInitializing environment with human rendering...")
    env = gym.make("CartPole-v1", render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 3. Instantiate the Agent and load the trained weights
    agent = DQNAgent(state_dim, action_dim, device)
    
    print(f"Loading weights from {WEIGHTS_PATH}...")
    try:
        agent.policy_net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
        agent.policy_net.eval()  # Set network to evaluation mode (turns off dropout, batchnorm, etc.)
    except Exception as e:
        print(f"\nError loading weights: {e}")
        print("Make sure your DQN network architecture in src/agent.py matches the saved weights.")
        return

    print("\nStarting evaluation episodes (Epsilon = 0.0)...")
    eval_rewards = []
    
    for episode in range(NUM_EVAL_EPISODES):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # During evaluation, we want pure exploitation (epsilon = 0.0)
            action = agent.select_action(state, epsilon=0.0)
            
            # Step environment
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            
            # Introduce a tiny sleep delay so the human eye can actually track the animation
            time.sleep(0.02)
            env.render()
        eval_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}/{NUM_EVAL_EPISODES} -> Total Reward: {total_reward}")

    env.close()
    
    # 4. Print final summary statistics
    mean_score = np.mean(eval_rewards)
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Episodes evaluated: {NUM_EVAL_EPISODES}")
    print(f"Average Reward:     {mean_score:.2f} / 500.0")
    
    if mean_score >= 195.0:
        print("Status:             PASSED (Agent successfully balances the pole)")
    else:
        print("Status:             FAILED (Agent requires further hyperparameter tuning)")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
