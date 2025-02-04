@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Adjust forward velocity reward weighting
    forward_velocity_scale = 2.0  # Increase to emphasize impact early on
    forward_velocity_temp = 0.5   # New temperature for non-linear scaling
    forward_velocity_reward = forward_velocity_scale * torch.exp(forward_velocity_temp * forward_velocity)

    # Energy penalty enhancement
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.5  # Revised to moderate the penalty appropriately
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)

    # Combine components with exponential scaling for task score
    overall_temp = 0.2  # Adjusted to balance total reward transformation
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
