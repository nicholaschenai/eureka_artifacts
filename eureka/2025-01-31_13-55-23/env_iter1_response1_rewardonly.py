@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity with scaling to emphasize faster speeds
    forward_temp = 0.5
    forward_velocity_reward = torch.exp(forward_temp * forward_velocity) - 1.0
    
    # New formulation for energy penalty with scaling to increase variability
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.05  # Lowered temperature increases the dispersion of penalty
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
