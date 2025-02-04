@torch.jit.script
def compute_reward(velocity: torch.Tensor, 
                   actions: torch.Tensor, 
                   heading_proj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for forward velocity
    forward_velocity_reward = velocity[:, 0]  # Assuming index 0 corresponds to the forward direction
    
    # Temperature parameter for velocity, used in normalization
    vel_temperature = 0.1
    forward_velocity_reward = torch.exp(forward_velocity_reward / vel_temperature)
    
    # Reward for maintaining a forward direction
    direction_reward = heading_proj  # Reward for projecting heading in positive direction
    dir_temperature = 0.5
    direction_reward = torch.exp(direction_reward / dir_temperature)
    
    # Penalty for high actions (energy efficiency)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    action_temperature = 1.0
    action_penalty_exp = torch.exp(-action_penalty / action_temperature)
    
    # Total reward
    total_reward = forward_velocity_reward + direction_reward + action_penalty_exp
    
    # Return the total reward and each component for analysis
    reward_components = {
        'forward_velocity_reward': forward_velocity_reward,
        'direction_reward': direction_reward,
        'action_penalty': action_penalty_exp
    }
    
    return total_reward, reward_components
