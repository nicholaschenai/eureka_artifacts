@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Adjusted rescaling to further enhance learning
    velocity_temp = 0.15  # Slightly increased temperature for added sensitivity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Further refine the stability reward by increasing its responsiveness
    stability_reward = torch.abs(up_vec[:, 2] - 1.0)
    stability_temp = 15.0  # Enhanced temperature to boost discrimination
    transformed_stability_reward = torch.exp(-stability_reward * stability_temp)  # Stronger encouragement for upright posture

    # Balance rewards to focus more on running consistently and quickly
    total_reward = 2.0 * transformed_velocity_reward + 0.1 * transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
