@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate velocity reward by scaling potentials change
    velocity_reward = (potentials - prev_potentials) * 0.1

    # Adjust stability reward to have more impact by scaling
    stability_reward = 2.0 * (up_vec[:, 2] - 0.5)  # Scales to center around zero with emphasis

    # Combine the transformed rewards
    total_reward = velocity_reward + stability_reward

    # More transformation for a balanced range and explorability by RL
    velocity_temp = 0.5  # Lower temperature to reduce dominance
    stability_temp = 1.0  # Increase the impact of stability

    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0
    transformed_stability_reward = torch.exp(stability_reward * stability_temp) - 1.0

    # Rebalance the total reward
    total_reward = 0.8 * transformed_velocity_reward + 0.2 * transformed_stability_reward

    # Return individual reward components
    reward_dict = {
        "velocity_reward": transformed_velocity_reward,
        "stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
