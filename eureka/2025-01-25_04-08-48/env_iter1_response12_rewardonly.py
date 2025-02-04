@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute velocity reward: encouraging fast movement towards the target
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.05  # Adjusted temperature for scaling down high velocity rewards
    transformed_velocity_reward = torch.tanh(velocity_reward * velocity_temp)

    # Improve stability reward: encourage upright posture more aggressively
    stability_temp = 10.0  # Increased temperature to boost influence
    transformed_stability_reward = torch.exp(up_vec[:, 2] * stability_temp) - 1.0

    # Combine the transformed components with adjusted scaling factors
    total_reward = transformed_velocity_reward + 0.5 * transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
