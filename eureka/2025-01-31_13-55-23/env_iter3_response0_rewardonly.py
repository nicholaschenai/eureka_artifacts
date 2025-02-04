@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Normalize forward velocity to prevent excessive magnitudes
    forward_velocity_scale = 0.1  # New lower scale to normalize range
    forward_velocity_reward = forward_velocity_scale * torch.exp(forward_velocity)

    # Re-scale energy penalty for better optimization
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.5  # Decreased magnitude for smoother optimization
    energy_penalty_scaled = -energy_temp * torch.exp(energy_penalty)

    # Combine rewards using transformation
    overall_temp = 1.0
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
