# Assignment 3: Deep Q-Networks (DQN) | DeepBluffing

Hey everyone! Welcome to Assignment 3. 

In your previous tasks, you used Tabular Value Iteration and Policy Iteration to solve Frozen Lake. That worked perfectly because Frozen Lake has a small, discrete grid of 16 exact states. But in the real world—whether you are training an agent to control a physical system, an autonomous vehicle, or a complex multi-player card game—states are usually continuous. You can't build a lookup table for infinite possibilities.

This assignment bridges that gap. You will move from exact tables to **Function Approximation** by building a **Deep Q-Network (DQN)** to solve a classic continuous reinforcement learning benchmark: **CartPole-v1**.

---

## The Goal

Your mission is to train a neural network that learns how to balance a pole on a moving cart. The environment keeps track of your performance, and it is considered fully solved when your agent manages to maintain a **trailing 100-episode average score of 195.0 or higher**.

### The Environment Mechanics:
* **State Space:** 4 continuous values tracking the cart position, cart velocity, pole angle, and pole angular velocity.
* **Action Space:** 2 discrete choices—push the cart left (0) or push the cart right (1).

---

## Project Architecture

The repository is split into a core source module and helper execution scripts:

```text
├── src/
│   ├── __init__.py      # Handles internal module exports (Do not modify)
│   ├── agent.py         # TODO: DQN architecture & epsilon-greedy logic
│   ├── buffer.py        # TODO: Experience Replay Buffer memory
│   └── learning.py      # TODO: Bellman Optimality Loss computation
├── train.py             # TODO: Hyperparameters & core training loop wire-up
└── evaluate.py          # Visualizer script to watch your trained model perform
```
## Your Tasks

You need to open up the files and fill in the missing logic inside the designated `# TODO` blocks.

### 1. `train.py`
* **TODO 0 (Hyperparameters):** This is where a massive chunk of RL learning happens. You need to choose and tune your batch size, learning rate, memory capacity, target network update frequency, and epsilon decay parameters. We left some broad suggestions in the comments, but finding the exact sweet spot that stabilizes training is on you.
* **TODO 1 to 5:** Complete the training loop pipeline. You will grab actions from your agent, step the environment, save those experiences to the buffer, call the optimization function, and make sure the target network updates its weights at the right intervals.

### 2. `src/agent.py`
* **TODO 1 & 2:** Build a classic Multi-Layer Perceptron (MLP) using PyTorch network layers. Map out the structure and write the forward pass to return the raw, unactivated Q-values for each action.
* **TODO 3:** Code the epsilon-greedy selection rule. This wrapper needs to seamlessly swap between exploring the environment uniformly at random and exploiting your network's best predicted moves.

### 3. `src/buffer.py`
* **TODO 1 & 2:** Create the experience replay storage memory. You need to implement basic data insertion and random batch sampling using Python collections so the training pipeline can pull independent experiences.

### 4. `src/learning.py`
* **TODO 1 to 5:** This is the mathematical core of the DQN. You will unpack the sample batches, pull out the relevant Q-values for actions taken, compute the expected Bellman updates, evaluate the Mean Squared Error loss against the target network, and run the PyTorch backpropagation steps.

---

## Getting Started & Execution

### 1. Environment Setup
Make sure you have your virtual environment activated, then install the necessary dependencies:

```bash
pip install -r requirements.txt
```
### 2. Run Training
Once you have written code across all your TODOs, boot up the training pipeline:

```Bash
python train.py
```
Note: The script is configured to automatically stop and dump your optimal neural network weights into weights/dqn_cartpole.pth the moment your running average crosses the 195 mark.
### 3. Visual Evaluation
After saving your weights, test how well your agent balances the pole in real time by executing the evaluation visualizer locally:

```Bash
python evaluate.py
```
## Submission Checklist
Verify that your training run successfully converges and creates the weights/dqn_cartpole.pth file.

Commit your updated code blocks and the saved weights file back to your assigned repository:
```bash
git add .
git commit -m "Completed Assignment-3"
git push origin main
```
Check your GitHub repository link online to confirm all modifications are safely pushed before the deadline. Good luck!