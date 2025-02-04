@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = potentials - prev_potentials
    
    # Temper transformation
    velocity_temp = 0.3  # Decrease temperature to rescale more subtly
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Enhanced stability reward
    consistency_temp = 3.0  # Increased temperature for sensitivity
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * consistency_temp)

    # Improved aggregate total reward function to emphasize stability
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
