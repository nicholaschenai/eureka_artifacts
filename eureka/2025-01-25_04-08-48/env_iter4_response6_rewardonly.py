@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted velocity reward component
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1  # Decreased temperature for finer scaling
    transformed_velocity_reward = torch.tanh(velocity_reward * velocity_temp)

    # Uprightness reward revised
    stability_temp = 5.0  # Adjusting temperature for uprightness stability
    uprightness_reward = 1.0 - torch.abs(up_vec[:, 2] - 1.0)
    scaled_uprightness_reward = torch.exp(-(1.0 - uprightness_reward) * stability_temp)

    # Total reward re-balance to emphasize upright stability
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * scaled_uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "scaled_uprightness_reward": scaled_uprightness_reward
    }

    return total_reward, reward_dict
