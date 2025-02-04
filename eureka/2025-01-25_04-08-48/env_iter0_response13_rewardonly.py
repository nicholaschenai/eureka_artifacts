@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_force: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant variables
    velocity = root_states[:, 7:10]  # Linear velocity of the torso
    forward_velocity = velocity[:, 0]  # Assuming the x-direction is forward

    # Reward for forward velocity
    forward_velocity_reward = forward_velocity
    
    # Penalty for energy usage (sum of squared torques)
    energy_usage_penalty = torch.sum(dof_force ** 2, dim=-1) * dt

    # Penalty for jerky actions (smoothness)
    smoothness_penalty = torch.sum((actions[:, 1:] - actions[:, :-1]) ** 2, dim=-1)

    # Normalize individual components
    temperature_velocity_reward = 1.0
    temperature_energy_penalty = 0.01
    temperature_smoothness_penalty = 0.01

    normalized_velocity_reward = torch.exp(forward_velocity_reward / temperature_velocity_reward)
    normalized_energy_penalty = torch.exp(-energy_usage_penalty / temperature_energy_penalty)
    normalized_smoothness_penalty = torch.exp(-smoothness_penalty / temperature_smoothness_penalty)

    # Combine components into total reward
    total_reward = normalized_velocity_reward + normalized_energy_penalty + normalized_smoothness_penalty

    # Create a dictionary of individual reward components
    reward_components = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_usage_penalty": energy_usage_penalty,
        "smoothness_penalty": smoothness_penalty
    }

    return total_reward, reward_components
