@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Reduced scale for forward velocity.
    forward_velocity_scale = 0.3
    forward_velocity_reward = forward_velocity_scale * forward_velocity
    
    # Adjust penalty for energy to make it more impactful.
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Reduce energy temperature slightly for more impact.
    energy_temp = 0.7
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Consider a penalty for deviation to help with stability
    stability_penalty_temp = 0.01
    deviation_penalty = torch.norm(velocity[:, 1:], p=2, dim=-1)  # Penalizing sideways velocity
    stability_penalty_scaled = torch.exp(-stability_penalty_temp * deviation_penalty)

    # Total reward calculation with the new components
    total_reward = forward_velocity_reward + energy_penalty_scaled + stability_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stability_penalty_scaled": stability_penalty_scaled
    }

    return total_reward, reward_dict
