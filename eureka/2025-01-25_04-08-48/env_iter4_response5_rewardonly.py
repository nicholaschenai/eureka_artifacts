@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rescaled velocity reward component
    velocity_reward = potentials - prev_potentials
    
    # Rescale the velocity transformation
    velocity_temp = 0.15  # Reduced temperature for finer scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Adjusted stability/uprightness reward
    stability_temp = 6.0  # Increased temperature for greater sensitivity
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * stability_temp)

    # Adjust the total reward balance to emphasize stability more
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
