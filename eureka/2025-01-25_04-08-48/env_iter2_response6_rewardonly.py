@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Rescale velocity reward to enhance its effect
    velocity_temp = 0.15  # Adjusted temperature parameter for velocity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Introduce a new stability reward focusing on maintaining upright posture
    stability_penalty = (up_vec[:, 2] - 1.0).abs()
    stability_temp = 20.0  # Increased temperature parameter for sharper gradient
    transformed_stability_reward = torch.exp(-stability_penalty * stability_temp)  # Encourage upright posture

    # Introduce a penalty for deviation from a desired running height or angle
    height = root_states[:, 2]
    desired_height = 1.0  # Hypothetical desired height/altitude
    height_penalty = (height - desired_height).abs()
    height_temp = 15.0
    transformed_height_penalty = torch.exp(-height_penalty * height_temp)

    # Aggregate rewards, ensuring a balanced yet velocity-focused reward
    total_reward = 1.5 * transformed_velocity_reward + 0.5 * transformed_stability_reward + 0.2 * transformed_height_penalty

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward,
        "transformed_height_penalty": transformed_height_penalty
    }

    return total_reward, reward_dict
