@torch.jit.script
def compute_reward(velocity: torch.Tensor, actions: torch.Tensor, heading_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Ensure inputs are tensors on the same device
    device = velocity.device
    
    # Reward for forward velocity (assuming the forward direction is along the x-axis)
    forward_velocity_reward = velocity[:, 0]  # Taking x-component of velocity
    temp_forward_velocity = 1.0
    forward_velocity_reward = torch.exp(forward_velocity_reward / temp_forward_velocity) - 1.0
    
    # Penalty for excessive joint actions
    action_penalty = torch.sum(actions**2, dim=-1)
    temp_action_penalty = 0.1
    action_penalty = -torch.exp(action_penalty / temp_action_penalty)
    
    # Penalty for deviation from desired heading direction
    heading_penalty = 1.0 - heading_proj  # Reward is higher when heading_proj is closer to 1 indicating proper heading
    temp_heading_penalty = 0.1
    heading_penalty = -torch.exp(heading_penalty / temp_heading_penalty)
    
    # Total reward is the sum of the components
    total_reward = forward_velocity_reward + action_penalty + heading_penalty

    # Returning total reward and each sub-component in a dictionary
    reward_dict = {
        'forward_velocity_reward': forward_velocity_reward,
        'action_penalty': action_penalty,
        'heading_penalty': heading_penalty
    }
    
    return total_reward, reward_dict
