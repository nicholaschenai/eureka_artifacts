@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward scaled for better balance with other components
    velocity_scale = 0.01
    velocity_reward = (potentials - prev_potentials) * velocity_scale
    
    # Stability component enhanced with a quadratic form for better sensitivity
    stability_reward = (up_vec[:, 2] - 1.0) ** 2  # Penalize deviation from upright

    # Temperature controlled transformation for adjusted reward influence
    velocity_temp = 0.5  # Adjusted temperature for velocity reward
    stability_temp = 1.0  # Higher value to emphasize stability

    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    transformed_stability_reward = torch.exp(-stability_reward * stability_temp)  # Reward upright posture

    # Include balance by adding a living reward to further encourage stability
    living_reward = 0.1 * torch.ones_like(transformed_stability_reward)

    # Final reward as a weighted sum
    total_reward = transformed_velocity_reward + transformed_stability_reward + living_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": transformed_stability_reward,
        "living_reward": living_reward
    }

    return total_reward, reward_dict
