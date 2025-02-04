@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, up_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the velocity reward
    velocity_reward = potentials - prev_potentials
    
    # Increase the temperature slightly to encourage faster speed
    velocity_temp = 0.15  
    transformed_velocity_reward = torch.exp(velocity_reward * velocity_temp) - 1.0

    # Revise stability reward to focus on keeping the torso upright
    stability_measure = (1 - up_vec[:, 2]).abs()  # Using absolute deviation from upright position
    stability_temp = 20.0  # Adjusted to give more weight in penalty if not upright
    transformed_stability_reward = torch.exp(-stability_measure * stability_temp)

    # Add a new component for maintaining a forward direction
    forward_reward = torch.clamp(torch.sum(root_states[:, 7:9] ** 2, dim=-1), max=1.0)  # Encouraging forward velocity
    forward_temp = 5.0  # New temperature for this component to guide direction
    transformed_forward_reward = torch.exp(forward_reward * forward_temp) - 1.0

    # Aggregate rewards, adjusting weights for improved task success
    total_reward = 1.2 * transformed_velocity_reward + 0.1 * transformed_stability_reward + 0.5 * transformed_forward_reward

    # Collect reward components for diagnostics/analysis
    reward_dict = {
        "transformed_velocity_reward": transformed_velocity_reward,
        "transformed_stability_reward": transformed_stability_reward,
        "transformed_forward_reward": transformed_forward_reward
    }

    return total_reward, reward_dict
