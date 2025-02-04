@torch.jit.script
def compute_reward(velocity: torch.Tensor, 
                   up_proj: torch.Tensor, 
                   vel_loc: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Reward for forward velocity
    forward_velocity_reward = velocity[:, 0]
    
    # Penalize deviation from upright pose
    upright_penalty = (1.0 - up_proj).abs()
    
    # Encourage movement in the forward direction
    direction_penalty = torch.abs(vel_loc[:, 1])
    
    # Combine rewards and penalties
    total_reward = forward_velocity_reward - upright_penalty - direction_penalty
    
    # Temperature parameters for transformations
    tv_forw = 1.0  # Temperature for forward velocity reward
    tu_up = 1.0  # Temperature for upright penalty
    td_dir = 1.0  # Temperature for direction penalty

    # Transform rewards and penalties
    forward_velocity_reward = torch.exp(forward_velocity_reward / tv_forw)
    upright_penalty = torch.exp(-upright_penalty / tu_up)
    direction_penalty = torch.exp(-direction_penalty / td_dir)
    
    # Re-compute total reward with transformed components
    total_reward_transformed = forward_velocity_reward * upright_penalty * direction_penalty

    return total_reward_transformed, {
        'forward_velocity_reward': forward_velocity_reward,
        'upright_penalty': upright_penalty,
        'direction_penalty': direction_penalty
    }
