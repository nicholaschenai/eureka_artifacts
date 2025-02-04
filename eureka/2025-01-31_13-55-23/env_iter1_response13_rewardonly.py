@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity with a slight exponential boost for higher speeds
    velocity_temp = 0.5
    forward_velocity_reward = torch.exp(velocity_temp * forward_velocity) - 1.0
    
    # Linear penalty for energy usage
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Rescale energy penalty for more sensitivity
    energy_temp = 0.01  # Reduced temperature for more gradient
    energy_penalty_scaled = energy_temp * energy_penalty
    
    # Total reward calculation
    total_reward = forward_velocity_reward - energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
