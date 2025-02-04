@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Rescale velocity_reward with adjusted temperature for better impact balance
    velocity_temp = 0.05  # Further reduced temperature parameter for velocity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Redefine stability to consider how upright the humanoid remains
    stability_reward = (up_vec[:, 2] - 1.0).abs()  # Deviation from upright posture
    stability_temp = 15.0  # Increased temperature for greater sensitivity
    transformed_stability_reward = torch.exp(-stability_reward * stability_temp)

    # Introduce consistency in forward velocity reward
    target_velocity = 1.0  # Assume desired forward velocity is 1.0 m/s
    forward_velocity = velocity[:, 0]  # Consider only the forward component
    velocity_consistency_reward = -torch.abs(forward_velocity - target_velocity)
    consistency_temp = 0.5
    transformed_consistency_reward = torch.exp(velocity_consistency_reward * consistency_temp) - 1.0

    # Aggregate rewards with priorities on velocity and consistency
    total_reward = 1.0 * transformed_velocity_reward + 0.2 * transformed_stability_reward + 0.5 * transformed_consistency_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward,
        "transformed_consistency_reward": transformed_consistency_reward
    }

    return total_reward, reward_dict
