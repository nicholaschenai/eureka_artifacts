@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward motion is along the x-axis

    # Reward for forward velocity, possibly scaled with a higher emphasis on progress
    forward_velocity_temp = 0.2
    forward_velocity_reward = torch.exp(forward_velocity_temp * forward_velocity) - 1
    
    # Adjust energy penalty for better differentiation
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_penalty_temp = 0.8  # Increase penalty influence for better optimization
    energy_penalty_scaled = -torch.exp(energy_penalty_temp * energy_penalty * dt)

    # Total reward considering both forward progress and energy usage
    total_reward = forward_velocity_reward + energy_penalty_scaled

    # Return each individual component
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
