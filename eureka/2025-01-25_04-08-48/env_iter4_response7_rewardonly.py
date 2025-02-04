@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = potentials - prev_potentials
    
    # Rescale transformed velocity reward
    velocity_temp = 0.2
    transformed_velocity_reward = (torch.exp(velocity_reward * velocity_temp) - 1.0) * 0.5

    # Uprightness reward enhancement (rewritten to penalize deviation)
    upright_temp = 2.0
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * upright_temp)

    # Penalty for roll or pitch to encourage upright orientation
    roll_pitch_penalty = 0.1 * torch.abs(root_states[:, 3]) + torch.abs(root_states[:, 4])

    # Total reward calculated with rebalanced weighting
    total_reward = 0.5 * transformed_velocity_reward + 0.3 * uprightness_reward - roll_pitch_penalty

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward,
        "roll_pitch_penalty": roll_pitch_penalty
    }

    return total_reward, reward_dict
