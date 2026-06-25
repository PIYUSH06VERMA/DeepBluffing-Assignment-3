import torch
import torch.nn.functional as F

def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    """
    Performs one step of optimization on the policy network using the Bellman equation.
    """
    # Don't train if the buffer doesn't have enough data yet
    if len(memory) < batch_size:
        return

    # Sample a batch from memory
    transitions = memory.sample(batch_size)
    
    # Unpack the batch and convert to PyTorch Tensors
    # (Provided to ensure correct tensor dimensions: [batch_size, 1])
    batch = list(zip(*transitions))
    
    state_batch = torch.FloatTensor(batch[0]).to(device)
    action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(batch[3]).to(device)
    done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

    # ---------------------------------------------------------
    # TODO 1: Compute current Q-values.
    # Pass 'state_batch' through 'policy_net' and use 'gather' to find the Q-values 
    # corresponding to the actions in 'action_batch'.
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # TODO 2: Compute max next Q-values.
    # Pass 'next_state_batch' through 'target_net'. Detach the result from the graph, 
    # and find the maximum Q-value for each state.
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # TODO 3: Compute the Bellman target.
    # Formula: expected_Q = reward + gamma * max_next_Q * (1 - done)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # TODO 4: Compute the Loss.
    # Use Mean Squared Error (or Huber Loss) between current Q-values and the expected target.
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # TODO 5: Optimize the model.
    # 1. Zero the gradients.
    # 2. Backpropagate the loss.
    # 3. Step the optimizer.
    # ---------------------------------------------------------
    
    pass