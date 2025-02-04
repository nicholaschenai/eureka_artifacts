@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials

    # Adjust velocity reward with a temperature parameter for normalization
    velocity_temp = 0.2  # Lower temperature to reduce the impact of high velocity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Introduce a new way of calculating the stability reward by emphasizing consistency in the upward direction
    # Normalize up_vec to scale stability
    up_stability = up_vec[:, 2] - 1.0
    stability_temp = 0.5  # Modify temperature for fine-tuning impact
    transformed_stability_reward = torch.exp(-up_stability.abs() * stability_temp) 

    # Combine into total reward
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * transformed_stability_reward  # More balanced weights

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
