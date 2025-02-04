@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the normalized velocity reward by scaling it appropriately
    velocity_reward_raw = potentials - prev_potentials
    velocity_temp = 0.01  # Decrease the temperature to reduce the scale
    transformed_velocity_reward = torch.exp(velocity_reward_raw * velocity_temp) - 1.0

    # Enhance the stability reward by increasing the scale
    # Incentivize staying upright more strongly
    stability_temp = 1.0  # Increase temperature to make uprightness more critical
    stability_incentive = torch.where(up_vec[:, 2] > 0.3, up_vec[:, 2], torch.tensor(0.0, device=up_vec.device))
    transformed_stability_reward = torch.exp(stability_incentive * stability_temp) - 1.0

    # Final reward is a balanced sum of the modified components
    total_reward = transformed_velocity_reward + transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
