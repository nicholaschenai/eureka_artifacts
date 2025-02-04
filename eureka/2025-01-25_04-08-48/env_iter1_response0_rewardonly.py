@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, torso_rotation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity-based reward component, scaled down
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1  # Reduced temperature to scale down the reward
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Stability reward: encouraging upright with adjustment for angle
    stability_temp = 10.0  # Increased to enhance sensitivity
    torso_angle_deviation = torch.abs(1.0 - up_vec[:, 2])
    stability_reward = torch.exp(-torso_angle_deviation * stability_temp)

    # Balance reward, encouraging minimal torso rotation
    angle_penalty_temp = 5.0  # Additional component penalty for rotation
    torso_balance_reward = torch.exp(-torch.norm(torso_rotation[:, 1:3], p=2, dim=-1) * angle_penalty_temp)

    # Combine rewards
    total_reward = transformed_velocity_reward + stability_reward + torso_balance_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": stability_reward,
        "torso_balance_reward": torso_balance_reward
    }

    return total_reward, reward_dict
