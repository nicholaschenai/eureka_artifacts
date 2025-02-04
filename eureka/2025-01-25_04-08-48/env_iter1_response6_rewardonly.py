@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rescale velocity reward for balance
    velocity_reward = (potentials - prev_potentials) / 10.0
    
    # Introduce a penalty for not maintaining a minimum torso height, encouraging upright navigation
    min_torso_height = 1.0  # Assumed minimum height for balance
    torso_height_penalty = torch.where(root_states[:, 2] < min_torso_height, -1.0, 0.0)
    
    # Combine the transformed velocity reward with the stability penalty
    total_reward = velocity_reward + torso_height_penalty
    
    # Temperature Parameters for Transformation
    velocity_temp = 0.1  # Temperature for velocity reward
    fall_penalty_temp = 1.0  # Temperature for torso height penalty
    
    # Transform the rewards
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    transformed_torso_height_penalty = torch.exp(torso_height_penalty * fall_penalty_temp) - 1.0

    # Final combined reward
    total_reward = transformed_velocity_reward + transformed_torso_height_penalty

    # Return each component for diagnostics
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "torso_height_penalty": transformed_torso_height_penalty
    }

    return total_reward, reward_dict
