@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, angle_to_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity
    forward_velocity_temp = 0.5
    forward_velocity_reward = torch.exp(forward_velocity_temp * forward_velocity)
    
    # Penalty for high action magnitudes (to promote efficiency)
    energy_penalty_temp = 0.2
    energy_penalty_scaled = torch.exp(-energy_penalty_temp * torch.sum(actions**2, dim=-1))
    
    # Reward for maintaining alignment/heading to encourage stability
    stability_reward_temp = 0.1
    stability_reward = torch.exp(-stability_reward_temp * angle_to_target)
    
    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled + stability_reward
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stability_reward": stability_reward
    }

    return total_reward, reward_dict
