@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward component based on forward velocity
    forward_velocity_reward = forward_velocity
    
    # Penalty for high action magnitudes (to promote efficiency)
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Adjust the scaling of the energy penalty to have a more significant impact
    energy_temp = 0.5
    energy_penalty_transformed = torch.exp(-energy_temp * energy_penalty)
    
    # Combine the components for total reward
    velocity_scale = 1.0  # Scaling factor to adjust contribution
    energy_scale = 0.1    # Scaling factor to balance with velocity
    total_reward = (velocity_scale * forward_velocity_reward) + (energy_scale * energy_penalty_transformed)
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_transformed": energy_penalty_transformed
    }

    return total_reward, reward_dict
