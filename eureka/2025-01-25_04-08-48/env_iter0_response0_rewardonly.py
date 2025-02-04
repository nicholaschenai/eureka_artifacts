@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # The speed component of the reward function
    velocity_reward = potentials - prev_potentials

    # Encouraging the humanoid to stay upright by using the up vector's z-component
    stability_reward = up_vec[:, 2]

    # Combine the velocity reward and the stability reward
    total_reward = velocity_reward + stability_reward

    # Optional transformation to control the influence of each reward component
    velocity_temp = 1.0  # Temperature parameter for velocity reward
    stability_temp = 0.1  # Temperature parameter for stability reward

    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    transformed_stability_reward = torch.exp(stability_reward * stability_temp) - 1.0

    # Final reward as a weighted sum of transformed components
    total_reward = transformed_velocity_reward + transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
