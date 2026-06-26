import sys
import os
import gymnasium as gym
import torch
import numpy as np

from src import DQNAgent

def run_autograder():
    print("--- DeepBluffing Assignment 3 Autograder ---")
    weights_path = "weights/dqn_cartpole.pth"
    
    # 1. Check if the file was actually submitted
    if not os.path.exists(weights_path):
        print(f"[FAIL] Missing weights file: '{weights_path}'")
        print("Did you forget to commit and push your weights folder?")
        sys.exit(1)

    # 2. Initialize environment and agent
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cpu") # Force CPU for GitHub Actions
    
    try:
        agent = DQNAgent(state_dim, action_dim, device)
    except Exception as e:
        print(f"[FAIL] Could not initialize DQNAgent: {e}")
        sys.exit(1)

    # 3. Load the submitted weights
    try:
        agent.policy_net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        agent.policy_net.eval()
    except Exception as e:
        print(f"[FAIL] Could not load weights into the network: {e}")
        print("Your neural network architecture in src/agent.py does not match your saved weights.")
        sys.exit(1)

    # 4. Evaluate performance (Pure Exploitation: Epsilon = 0.0)
    print("Running 20 evaluation episodes...")
    eval_rewards = []
    
    for episode in range(20):
        state, _ = env.reset()
        total_reward = 0
        done, truncated = False, False
        
        while not (done or truncated):
            try:
                action = agent.select_action(state, epsilon=0.0)
                
                # Ensure the action is passed as a standard python integer, not a raw tensor
                if torch.is_tensor(action):
                    action = action.item()
                    
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
            except Exception as e:
                print(f"[FAIL] Crash during environment stepping: {e}")
                sys.exit(1)
                
        eval_rewards.append(total_reward)

    env.close()

    # 5. Final Grade Calculation
    mean_score = np.mean(eval_rewards)
    print(f"\nFinal Average Score: {mean_score:.2f} / 500.0")
    
    if mean_score >= 195.0:
        print("[PASS] The agent successfully solved CartPole!")
        sys.exit(0) # Exit 0 tells GitHub the test passed
    else:
        print(f"[FAIL] Score too low. Expected >= 195.0, got {mean_score:.2f}")
        sys.exit(1) # Exit 1 tells GitHub the test failed

if __name__ == "__main__":
    run_autograder()
