@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rescaled velocity reward to balance with stability reward
    velocity_temp = 0.01  # Adjusted temperature parameter for velocity reward
    velocity_reward = potentials - prev_potentials
    scaled_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Revised stability reward to also consider the uprightness and potential deviation
    stability_temp = 2.0  # Increased temperature parameter for stronger influence
    uprightness = up_vec[:, 2]
    stability_reward = uprightness * (1.0 - torch.abs(up_vec[:, 0]) - torch.abs(up_vec[:, 1]))  # Penalize horizontal deviation
    transformed_stability_reward = torch.exp(stability_reward * stability_temp) - 1.0

    # Total reward as a combination of rescaled components
    total_reward = scaled_velocity_reward + transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "scaled_velocity_reward": scaled_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
