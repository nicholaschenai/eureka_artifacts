@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Scale down the velocity reward
    velocity_temp = 0.1
    velocity_reward = potentials - prev_potentials
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Rewriting and enhancing stability reward
    stability_temp = 2.5  # Adjusted temperature for better sensitivity
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * stability_temp)

    # Aggregate total reward with adjusted weights
    total_reward = 0.5 * transformed_velocity_reward + 1.0 * uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
