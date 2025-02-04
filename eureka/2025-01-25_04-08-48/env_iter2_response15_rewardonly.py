@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Adjust temperature for better scaling
    velocity_temp = 0.08  # Reduced to moderate influence
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Re-design the stability reward to measure consistency in maintaining an upright position
    stability_metric = (((up_vec[:, 2] - 1.0) ** 2).mean()).sqrt()
    stability_temp = 15.0  # Increase to boost sensitivity and variability
    transformed_stability_reward = torch.exp(-stability_metric * stability_temp)

    # Aggregate rewards with balanced scaling
    total_reward = 0.8 * transformed_velocity_reward + 0.3 * transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
