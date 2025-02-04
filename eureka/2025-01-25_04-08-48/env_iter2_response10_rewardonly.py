@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Improved stability reward based on upright position and angular stability
    upright_penalty = torch.abs(up_vec[:, 2] - 1.0)  # Deviation from upright
    ang_velocity_penalty = torch.norm(ang_velocity, p=2, dim=-1)  # Penalize high angular velocity
    combined_stability_penalty = upright_penalty + 0.5 * ang_velocity_penalty  # Weight angular penalty

    stability_temp = 5.0  # Moderate temperature to balance influence
    transformed_stability_reward = torch.exp(-combined_stability_penalty * stability_temp)

    # Aggregate rewards, with a focus on speed but not ignoring stability
    total_reward = 1.0 * transformed_velocity_reward + 0.1 * transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
