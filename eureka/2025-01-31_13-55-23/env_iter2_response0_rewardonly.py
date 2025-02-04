@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Forward velocity with a marginally adjusted scaling factor
    velocity_scale = 0.6  # Slightly increased for more impact
    forward_velocity_reward = velocity_scale * forward_velocity
    
    # Rewritten energy penalty using a different transformation for more effective scaling and impact
    energy_penalty = torch.sum(actions**2, dim=-1)
    # Re-scaling with a logarithmic transformation to dampen large penalties and encourage smooth actions
    energy_temp = 1.0  
    energy_penalty_log_scaled = -torch.log(1 + energy_temp * energy_penalty)

    # Total reward calculation
    total_reward = forward_velocity_reward + energy_penalty_log_scaled
    
    # Individual reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_log_scaled": energy_penalty_log_scaled
    }

    return total_reward, reward_dict
