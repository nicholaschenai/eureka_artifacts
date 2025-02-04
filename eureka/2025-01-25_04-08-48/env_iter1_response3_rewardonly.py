@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute velocity-based reward component
    velocity_reward_raw = potentials - prev_potentials
    velocity_temp = 0.1  # Adjusting temperature for smoother scaling
    velocity_reward = torch.exp(velocity_reward_raw * velocity_temp) - 1.0

    # Compute stability-based reward component using the uprightness measure
    stability_temp = 5.0  # Increasing weight on stability for strong upright signals
    stability_reward_raw = up_vec[:, 2].clamp(0, 1)  # Ensure it stays within [0, 1]
    stability_reward = stability_reward_raw * stability_temp

    # Combine rewards. We initially give a stronger influence to stability to better shape early learning.
    total_reward = 0.5 * velocity_reward + 2.0 * stability_reward

    # Final transformation to control the overall balance and reduce volatility
    final_temp = 0.01
    total_reward_transformed = torch.exp(total_reward * final_temp) - 1.0

    # Capture individual components for analysis
    reward_dict = {
        "velocity_reward": velocity_reward,
        "stability_reward": stability_reward
    }

    return total_reward_transformed, reward_dict
