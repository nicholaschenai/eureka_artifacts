@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # The speed component of the reward function, scaled down
    velocity_reward = (potentials - prev_potentials) * 0.1  # Adjust scaling factor

    # Revised calculation for the stability reward
    stability_temp = 5.0  # Adjust temperature to increase sensitivity
    stability_reward = torch.exp(up_vec[:, 2] * stability_temp) - 1.0

    # Add a penalty for deviation from uprightness if necessary
    upright_penalty = (1.0 - up_vec[:, 2]) * 0.5  # Penalty for not being upright, scale accordingly

    # Combine the velocity reward and the bonus/penalty for stability
    total_reward = velocity_reward + stability_reward - upright_penalty

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": velocity_reward,
        "stability_reward": stability_reward,
        "upright_penalty": upright_penalty,
    }

    return total_reward, reward_dict
