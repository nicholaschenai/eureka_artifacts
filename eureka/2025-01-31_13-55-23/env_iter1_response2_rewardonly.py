@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity
    forward_velocity_reward = forward_velocity
    
    # Penalty for high action magnitudes (to promote efficiency)
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Adjusted scaling for energy penalty
    energy_temp = 0.5  # Increased to make penalty more significant
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward with normalization
    velocity_temp = 0.1  # Temperature for scaling forward velocity
    forward_reward_scaled = torch.exp(velocity_temp * forward_velocity_reward)

    total_reward = forward_reward_scaled + energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "forward_reward_scaled": forward_reward_scaled,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
