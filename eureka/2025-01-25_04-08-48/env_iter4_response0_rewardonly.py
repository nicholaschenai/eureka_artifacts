@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rescale velocity reward component with reduced temperature
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1  # Further reduce temperature for subtle scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Adjust the uprightness reward for greater detail
    uprightness_temp = 10.0  # Increase to improve sensitivity
    up_vector_deviation = 1.0 - up_vec[:, 2]
    uprightness_reward = torch.exp(-up_vector_deviation * uprightness_temp)

    # Adjust weights to balance between speed and uprightness
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * uprightness_reward

    # Dictionary to track individual components
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
