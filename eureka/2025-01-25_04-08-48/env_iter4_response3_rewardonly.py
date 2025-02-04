@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = potentials - prev_potentials

    # Temperature for velocity reward
    velocity_temp = 0.1  # Decrease temperature for more subtle scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Uprightness reward with adjusted scale
    uprightness_temp = 10.0  # Higher temperature to increase sensitivity
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * uprightness_temp)

    # New stability reward penalizing high angular velocities
    ang_velocity_norm = torch.norm(ang_velocity, p=2, dim=-1)
    stability_penalty_temp = 1.0
    stability_penalty = torch.exp(-ang_velocity_norm * stability_penalty_temp)

    # Total reward
    total_reward = 0.5 * transformed_velocity_reward + 0.2 * uprightness_reward + 0.3 * stability_penalty

    # Reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward,
        "stability_penalty": stability_penalty
    }

    return total_reward, reward_dict
