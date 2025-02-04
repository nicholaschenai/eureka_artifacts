@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Rescale the velocity reward for better balance
    velocity_temp = 0.3  # Reduced temperature for a more moderate impact
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Redefine a more effective stability/consistency reward
    uprightness = torch.abs(up_vec[:, 2] - 1.0)
    consistency_temp = 3.0  # Higher temperature for stability sensitivity
    consistency_reward = torch.exp(-uprightness * consistency_temp) 

    # Add a smoothness reward to reduce erratic movements
    smoothness_temp = 0.1
    delta_potentials_norm = torch.norm(potentials - prev_potentials, p=2, dim=-1)
    smoothness_reward = torch.exp(-delta_potentials_norm * smoothness_temp)

    # Combine into total reward
    total_reward = 0.6 * transformed_velocity_reward + 0.3 * consistency_reward + 0.1 * smoothness_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "consistency_reward": consistency_reward,
        "smoothness_reward": smoothness_reward
    }

    return total_reward, reward_dict
