@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute velocity reward
    velocity_reward = potentials - prev_potentials

    # Rescale the velocity reward with a new temperature parameter
    velocity_temp = 0.1  # Reduced temperature to rescale the reward contribution
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Redefining the stability reward based on uprightness
    stability_temp = 5.0  # New temperature for significant variation
    uprightness_error = (1.0 - up_vec[:, 2]).clamp(min=0.0)
    stability = torch.exp(-uprightness_error * stability_temp)

    # Compute total reward
    total_reward = 0.6 * transformed_velocity_reward + 0.4 * stability  # Adjusted weights for balance

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "stability": stability
    }

    return total_reward, reward_dict
