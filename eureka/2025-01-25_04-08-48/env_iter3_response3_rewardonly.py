@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity-based progress reward
    velocity_reward = potentials - prev_potentials
    
    # Adjust velocity reward with a lower temperature parameter for normalization
    velocity_temp = 0.1  # Reduced temperature for better scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Redefine the consistency reward to incorporate more dynamic uprightness maintenance
    consistency_temp = 2.0  # Increased temperature for greater sensitivity
    uprightness_reward = torch.exp(-torch.abs(up_vec[:, 2] - 1.0) * consistency_temp) - 0.5

    # Normalize both rewards to a similar scale
    transformed_velocity_reward = transformed_velocity_reward / torch.max(transformed_velocity_reward).clamp_min(1.0)
    uprightness_reward = uprightness_reward / torch.max(uprightness_reward).clamp_min(1.0)

    # Calculate a combined total reward
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * uprightness_reward  # Adjusted weights for balance

    # Collect reward components for detailed diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
