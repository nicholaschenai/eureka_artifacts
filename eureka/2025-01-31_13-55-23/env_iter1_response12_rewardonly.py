@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity
    forward_velocity_reward = forward_velocity
    
    # Penalty for high action magnitudes (to encourage energy efficiency)
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Adjust scaling of the energy penalty
    energy_temp = 0.5
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Introduce a balance factor between forward movement and energy efficiency
    balance_factor = 0.1
    balanced_reward = forward_velocity_reward + balance_factor * energy_penalty_scaled
    
    # Ensure reward components contribute meaningfully
    forward_temp = 0.3
    forward_velocity_transformed = torch.exp(forward_temp * forward_velocity_reward)
    
    total_reward = forward_velocity_transformed + balance_factor * energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_transformed": forward_velocity_transformed,
        "energy_penalty_scaled": energy_penalty_scaled,
        "balanced_reward": balanced_reward
    }

    return total_reward, reward_dict
