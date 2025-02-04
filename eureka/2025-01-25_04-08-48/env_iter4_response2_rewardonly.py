@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = potentials - prev_potentials
    
    # Rescale velocity transformation
    velocity_temp = 0.2  # Further decrease the temperature for subtle scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    
    # Redesigned stability reward
    upright_deviation = torch.abs(1.0 - up_vec[:, 2])  # Absolute deviation from upright
    uprightness_temp = 5.0  # Increase temperature for more distinct differentiation
    uprightness_reward = torch.exp(-upright_deviation * uprightness_temp)
    
    # Balance weightings for combined reward
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
