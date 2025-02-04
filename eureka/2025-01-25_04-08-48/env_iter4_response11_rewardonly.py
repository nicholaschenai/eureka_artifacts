@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Velocity reward component
    velocity_reward = potentials - prev_potentials
    
    # Temperature parameters for transformations
    velocity_temp = 0.1  # Reduced temperature for subtle scaling
    uprightness_temp = 5.0  # Increased temperature to enhance differentiation

    # Transforming rewards with updated scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * uprightness_temp)

    # Adjust weightings for better balance emphasis
    total_reward = 0.4 * transformed_velocity_reward + 0.6 * uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
