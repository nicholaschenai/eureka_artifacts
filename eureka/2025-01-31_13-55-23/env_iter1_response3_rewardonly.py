@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity
    forward_velocity_reward = forward_velocity

    # Penalty for high action magnitudes (to promote efficiency)
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Rescale energy penalty with a different temperature
    energy_temp = 0.5
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)
  
    # Normalize forward velocity to a similar scale
    vel_temp = 0.1
    forward_velocity_reward_scaled = torch.exp(vel_temp * forward_velocity_reward)

    # Total reward combining the adjusted components
    total_reward = forward_velocity_reward_scaled + energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward_scaled": forward_velocity_reward_scaled,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
