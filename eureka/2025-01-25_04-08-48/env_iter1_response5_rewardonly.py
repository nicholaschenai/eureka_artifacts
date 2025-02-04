@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Rescale velocity_reward to have significant impact with less magnitude
    velocity_temp = 0.1  # Adjusted temperature parameter for velocity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Re-evaluate the stability reward using a better metric, such as deviation from upright position
    stability_reward = (up_vec[:, 2] - 1.0).abs()
    stability_temp = 10.0  # Increased temperature parameter for better gradient
    transformed_stability_reward = torch.exp(-stability_reward * stability_temp)  # Encourage upright posture

    # Aggregate rewards, giving more focus on velocity to drive the task of running
    total_reward = 1.0 * transformed_velocity_reward + 0.1 * transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
