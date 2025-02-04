@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Transformed Velocity Reward Component
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1  # Further decrease temperature to reduce its scaling effect
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Uprightness Reward Component
    uprightness_temp = 2.0  # Slightly increased to emphasize uprightness changes
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * uprightness_temp)

    # New Angular Velocity Penalty for Stability
    ang_vel_penalty_temp = 0.5  # Introduced to penalize angular velocity for instability
    ang_vel_stability_penalty = torch.exp(-torch.norm(ang_velocity, dim=1) * ang_vel_penalty_temp)

    # Aggregated Total Reward
    total_reward = 0.5 * transformed_velocity_reward + 0.4 * uprightness_reward + 0.1 * ang_vel_stability_penalty

    # Collect Reward Components for Diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward,
        "ang_vel_stability_penalty": ang_vel_stability_penalty
    }

    return total_reward, reward_dict
