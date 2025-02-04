@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component with adjusted scaling
    velocity_reward = potentials - prev_potentials
    
    # Reduced temper transformation for velocity
    velocity_temp = 0.1  # Decrease temperature to lessen its dominance
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Uprightness reward with revised scale
    uprightness_reward = up_vec[:, 2]
    uprightness_temp = 5.0  # Increased temperature to emphasize sensitivity
    transformed_uprightness_reward = torch.exp((uprightness_reward - 1.0) * uprightness_temp)

    # Balancing the total reward more towards uprightness
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * transformed_uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_uprightness_reward": transformed_uprightness_reward
    }

    return total_reward, reward_dict
