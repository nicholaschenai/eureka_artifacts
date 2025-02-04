@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor, ang_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Increase temperature parameter for velocity to maximize impact
    velocity_temp = 0.5  # Adjusted temperature parameter for velocity
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Refine the stability reward to consider angular velocity deviation from upright
    stability_reward = (up_vec[:, 2] - 1.0).abs() + torch.norm(ang_velocity, dim=1)
    stability_temp = 5.0  # Adjusted temperature for a more sensitive stability measure
    transformed_stability_reward = torch.exp(-stability_reward * stability_temp)  

    # Aggregate rewards with adjusted weights for improved balance
    total_reward = 0.7 * transformed_velocity_reward + 0.3 * transformed_stability_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward
    }

    return total_reward, reward_dict
