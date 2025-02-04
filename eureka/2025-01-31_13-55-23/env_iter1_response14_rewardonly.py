@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Transform forward velocity reward with a temperature parameter
    velocity_temp = 0.5
    forward_velocity_reward = torch.exp(velocity_temp * forward_velocity)
    
    # Penalty for action magnitudes (to promote efficiency)
    actions_squared = torch.sum(actions**2, dim=-1)
    
    # Tuning energy penalty with a different temperature
    energy_temp = 0.05
    energy_penalty_scaled = torch.exp(-energy_temp * actions_squared)
    
    # Total reward combination
    total_reward = forward_velocity_reward + energy_penalty_scaled

    # Reward components for tracking
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
