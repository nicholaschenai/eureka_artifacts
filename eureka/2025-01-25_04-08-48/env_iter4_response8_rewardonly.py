@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = potentials - prev_potentials
    
    # Adjusted transformation temperature
    velocity_temp = 0.2  # Further decreasing for balanced influence
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Rewritten enhanced uprightness reward
    orientation_deviation = 1.0 - up_vec[:, 2]
    uprightness_reward = torch.exp(-orientation_deviation * 5.0)  # Increased sensitivity

    # New high-speed maintenance reward
    speed_reward = torch.clamp(velocity_reward, min=0.0)  # Encourages maintaining high speeds

    # Balancing total reward with new components
    total_reward = 0.5 * transformed_velocity_reward + 0.3 * uprightness_reward + 0.2 * speed_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward,
        "speed_reward": speed_reward
    }

    return total_reward, reward_dict
