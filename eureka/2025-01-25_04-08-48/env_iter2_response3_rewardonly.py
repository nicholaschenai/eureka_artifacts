@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    velocity_temp = 0.2  # Adjusted temperature parameter for more balanced influence
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Enhance the stability reward by considering angular deviation
    stability_deviation = (up_vec[:, 2] - 1.0).abs()
    stability_temp = 20.0  # Increase temperature for higher sensitivity
    transformed_stability_reward = torch.exp(-stability_deviation * stability_temp)

    # Additional forward persistence reward
    forward_velocity = velocity[:, 0]  # Assuming the x-direction is forward
    forward_persistence_temp = 0.1   
    forward_persistence_reward = torch.exp(forward_velocity * forward_persistence_temp) - 1.0

    # Aggregate rewards with more focus on balancing velocity and stability, and encouraging consistent forward motion
    total_reward = 1.0 * transformed_velocity_reward + 0.5 * transformed_stability_reward + 0.2 * forward_persistence_reward

    # Collect reward components
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward,
        "forward_persistence_reward": forward_persistence_reward
    }

    return total_reward, reward_dict
