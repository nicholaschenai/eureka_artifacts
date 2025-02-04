@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, heading_vec: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Calculate the potential difference to encourage moving towards target
    potential_reward = potentials - prev_potentials

    # Calculate velocity in the forward direction
    velocity = root_states[:, 7:10]  # Extract velocity from root_states
    forward_velocity = velocity[:, 0]  # Assuming x-axis is forward; adjust if different

    # Reward for moving fast in the forward direction
    forward_reward = forward_velocity * dt
    
    # Regularization to discourage excessive action
    action_penalty_weight = 0.1
    action_penalty = -action_penalty_weight * torch.sum(actions**2, dim=-1)
    
    # Normalize rewards using exponential transformation
    potential_temp = 1.0
    forward_temp = 2.0
    penalty_temp = 0.5
    
    potential_reward_transformed = torch.exp(potential_temp * potential_reward)
    forward_reward_transformed = torch.exp(forward_temp * forward_reward)
    action_penalty_transformed = torch.exp(penalty_temp * action_penalty)
    
    # Total reward is a combination of these components
    reward = potential_reward_transformed + forward_reward_transformed + action_penalty_transformed
    
    # Store individual reward components for analysis and debugging
    reward_components = {
        "potential_reward": potential_reward,
        "forward_reward": forward_reward,
        "action_penalty": action_penalty,
        "potential_reward_transformed": potential_reward_transformed,
        "forward_reward_transformed": forward_reward_transformed,
        "action_penalty_transformed": action_penalty_transformed
    }
    
    return reward, reward_components
