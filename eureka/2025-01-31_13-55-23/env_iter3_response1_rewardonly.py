@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Adjusting the scaling factor for the forward velocity reward
    forward_velocity_scale = 1.0  # Reduced for better balance
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Increase the impact of the energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Apply a stronger penalty with increased influence 
    energy_temp = 3.0  # Increased for higher emphasis
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)

    # Total reward assembly 
    total_temp = 0.1  # Temperature parameter for normalization
    total_reward = torch.exp(total_temp * (forward_velocity_reward + energy_penalty_scaled))

    # Reward components for diagnostic purposes
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
