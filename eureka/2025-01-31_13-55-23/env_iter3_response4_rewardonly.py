@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Forward velocity reward scaling
    forward_velocity_temp = 0.2
    forward_velocity_reward = torch.exp(forward_velocity_temp * forward_velocity)

    # Revise energy penalty calculation
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.5  # Adjusted for better sensitivity and influence
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)  # Exponential form for strong penalization

    # Combine rewards to form total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled

    # Normalize total reward with its own temperature
    total_temp = 0.05
    total_reward_normalized = torch.exp(total_temp * total_reward) - 1  # Shifted to avoid negative total reward values

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
    }

    return total_reward_normalized, reward_dict
