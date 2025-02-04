@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Further scale down the velocity reward
    velocity_temp = 0.05  # Adjusted temperature parameter to further balance the rewards
    transformed_velocity_reward = (torch.exp(velocity_reward * velocity_temp) - 1.0).clamp(min=0)

    # Refine the stability reward component
    stability_offset = 0.8
    stability_reward = (up_vec[:, 2] - 1.0).abs() - stability_offset
    stability_temp = 15.0  # Increased temperature parameter for better gradient response
    transformed_stability_reward = torch.exp(-stability_reward * stability_temp)
    
    # Aggregate rewards, balancing them for improved optimization
    total_reward = 1.0 * transformed_velocity_reward + 0.2 * transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
