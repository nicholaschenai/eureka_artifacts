@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = root_states.device

    # Velocity reward based on potential difference
    velocity_reward = potentials - prev_potentials

    # Stability reward: Encourage upright posture, but increase the impact of deviations from upright
    stability_temp = 5.0  # Increased temperature to amplify the impact
    stability_reward = torch.exp(up_vec[:, 2] * stability_temp) - 1.0
    
    # Penalty for deviation from upright posture (additional component)
    alignment_temp = -10.0  # Encourage alignment by heavily penalizing misalignment
    alignment_penalty = torch.exp((1.0 - up_vec[:, 2]) * alignment_temp) - 1.0

    # Total reward combining velocity, stability, and alignment penalties
    total_reward = velocity_reward + stability_reward + alignment_penalty

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": velocity_reward,
        "stability_reward": stability_reward,
        "alignment_penalty": alignment_penalty
    }

    return total_reward, reward_dict
