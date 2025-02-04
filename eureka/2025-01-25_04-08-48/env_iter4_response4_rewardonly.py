@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity Reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.2  # Rescaled for reduced impact
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Enhanced Uprightness Reward
    uprightness_temp = 4.0  # Increased for higher sensitivity
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * uprightness_temp)

    # Penalty for excessive angular deviation
    angular_deviation_penalty = 1.0 - up_vec[:, 2]

    # Improved Total Reward Sum
    total_reward = 0.5 * transformed_velocity_reward + 0.4 * uprightness_reward - 0.1 * angular_deviation_penalty

    # Reward component tracking
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward,
        "angular_deviation_penalty": angular_deviation_penalty
    }

    return total_reward, reward_dict
