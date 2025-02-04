@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Adjust the velocity reward with a smaller temperature
    velocity_temp = 0.2  # Reduced temperature for less dominance
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Introduce a stability reward focusing on the torso's upright posture
    upright_temp = 0.8  # Adjusted temperature for upright sensitivity
    uprightness = (up_vec[:, 2] - 1.0).abs()
    stability_reward = torch.exp(-uprightness * upright_temp) 

    # Introduce a new reward for alignment and smoothness to aid stability
    alignment_temp = 1.0
    alignment_reward = torch.max(torch.tensor(1.0) - uprightness, torch.tensor(0.0)) ** alignment_temp

    # Combine into total reward with updated balance
    total_reward = 0.6 * transformed_velocity_reward + 0.2 * stability_reward + 0.2 * alignment_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "stability_reward": stability_reward,
        "alignment_reward": alignment_reward
    }

    return total_reward, reward_dict
