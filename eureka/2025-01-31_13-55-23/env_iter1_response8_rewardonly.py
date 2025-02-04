@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, max_velocity: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Normalize forward velocity reward
    velocity_temp = 0.5  # Reducing temperature for sharper differentiation
    forward_velocity_reward = torch.exp(forward_velocity / max_velocity * velocity_temp)
    
    # Penalty for high action magnitudes (to promote efficiency)
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Adjusting energy penalty scaling
    energy_temp = 0.5  # Increased temperature for more penalty influence
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
