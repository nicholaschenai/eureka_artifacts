@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.1  # Suitable temperature parameter
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Enhanced stability reward with greater sensitivity
    deviation_from_upright = torch.abs(up_vec[:, 2] - 1.0)
    stability_temp = 20.0  # Increased temperature for better gradient response
    transformed_stability_reward = torch.exp(-deviation_from_upright * stability_temp)

    # Introduce additional reward for consistent forward progress (not just velocity, but also direction)
    progress_temp = 0.3
    forward_movement_reward = torch.tanh(prev_potentials) * progress_temp

    # Aggregate rewards with enhanced scaling
    total_reward = (
        1.0 * transformed_velocity_reward +
        0.5 * transformed_stability_reward +
        0.1 * forward_movement_reward
    )

    # Collect reward components for diagnostics and analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward,
        "forward_movement_reward": forward_movement_reward
    }

    return total_reward, reward_dict
