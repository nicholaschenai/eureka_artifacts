@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rescaled velocity reward component to reduce the dominance of speed
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.15  # Decrease temperature further to rescale more subtly
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Enhanced stability through uprightness reward
    uprightness_temp = 5.0  # Increase temperature for sensitivity
    uprightness_reward = torch.exp(-(1.0 - up_vec[:, 2]) * uprightness_temp)

    # Revised total reward calculation to better balance speed vs balance
    total_reward = 0.5 * transformed_velocity_reward + 0.5 * uprightness_reward

    # Collect reward components for diagnostics
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "uprightness_reward": uprightness_reward
    }

    return total_reward, reward_dict
