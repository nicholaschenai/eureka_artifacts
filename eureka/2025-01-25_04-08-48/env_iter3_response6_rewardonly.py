@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Re-scale velocity reward with adjustments to prevent domination
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.3  # Adjusted to reduce impact
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Rewrite consistency reward to account for stable movement and orientation in the running direction
    stability_temp = 2.0  # Increased temperature for higher sensitivity
    uprightness_loss = ((up_vec[:, 2] - 1.0) ** 2).mean()  # Penalize deviation from upright position
    balanced_running_reward = torch.exp(-uprightness_loss * stability_temp)

    # Combine total reward with adjusted balance
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * balanced_running_reward  # Increasing influence of balance

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "balanced_running_reward": balanced_running_reward
    }

    return total_reward, reward_dict
