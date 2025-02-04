@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assume forward direction is along the x-axis

    # Reward for forward velocity with transformation
    velocity_temp = 0.5  # Temperature for velocity transformation
    forward_velocity_reward = torch.exp(velocity_temp * forward_velocity)

    # New form for energy penalty using a direct penalty
    energy_penalty_weight = 0.1  # Weight for energy penalty
    energy_penalty = energy_penalty_weight * torch.sum(actions**2, dim=-1)

    # Normalize energy penalty (more pronounced impact)
    energy_temp = 0.5
    energy_penalty_normalized = 1.0 - torch.exp(-energy_temp * energy_penalty)

    # Total reward
    total_reward = forward_velocity_reward - energy_penalty_normalized

    # Reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_normalized": energy_penalty_normalized
    }

    return total_reward, reward_dict
