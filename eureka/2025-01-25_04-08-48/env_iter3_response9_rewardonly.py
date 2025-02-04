@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.3  # Reduced temperature for re-scaling
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Introduce a novel upright reward for incentivizing stability
    stability_temp = 0.5  # Temperature for sensitivity
    uprightness_reward = torch.clamp(up_vec[:, 2], 0.0, 1.0)  # Directly using vector's Z direction for uprightness
    transformed_uprightness_reward = torch.exp(stability_temp * (uprightness_reward - 1.0))

    # Total reward is a combination of velocity, uprightness, and a favoring long episode runs
    total_reward = 0.7 * transformed_velocity_reward + 0.3 * transformed_uprightness_reward

    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_uprightness_reward": transformed_uprightness_reward
    }

    return total_reward, reward_dict
