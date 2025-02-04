@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Linear velocity
    forward_velocity = velocity[:, 0]  # Assuming forward is x-axis

    # Reward for forward velocity with a slightly higher scale
    velocity_scale = 1.0
    forward_velocity_reward = velocity_scale * forward_velocity
    
    # Increased temperature for a stronger penalty on energy usage
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0  # Further increasing to provide a stronger penalty
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward computation
    total_reward = forward_velocity_reward + energy_penalty_scaled
    
    # Reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
