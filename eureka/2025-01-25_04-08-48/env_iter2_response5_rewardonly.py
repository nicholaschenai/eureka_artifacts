@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Enhanced stability reward for better sensitivity to upright positioning
    stability_error = torch.sqrt(1 - up_vec[:, 2] ** 2)
    stability_temp = 5.0  # Lower than before, allowing for better optimization
    transformed_stability_reward = torch.exp(-stability_error * stability_temp)

    # Adjusting weights to strike balance between running fast and maintaining stability
    total_reward = 1.0 * transformed_velocity_reward + 0.3 * transformed_stability_reward

    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }
    
    return total_reward, reward_dict
