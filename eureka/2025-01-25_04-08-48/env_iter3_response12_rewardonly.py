@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Introduce a lower temperature for velocity to reduce its dominance
    velocity_temp = 0.2  # Reduced temperature for lower sensitivity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Revise and rescale the consistency reward using angle to uprightness
    upright_temp = 5.0  # New temperature for enhanced significance
    upright_reward = torch.exp(-upright_temp * (up_vec[:, 2] - 1.0).abs()) - 1.0

    # Combine into total reward with new weighting
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * upright_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "upright_reward": upright_reward
    }

    return total_reward, reward_dict
