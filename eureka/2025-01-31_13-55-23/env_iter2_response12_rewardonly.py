@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant data
    velocity = root_states[:, 7:10]  # Linear velocity of the torso
    forward_velocity = velocity[:, 0]  # Assuming forward direction is x-axis
    
    # Reward for forward velocity
    velocity_scale = 0.3
    forward_velocity_reward = velocity_scale * forward_velocity
    
    # Energy penalty with increased temperature for higher influence
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0  # Increased temperature for stronger impact
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)
    
    # Introduce additional efficiency component related to smoothness
    action_diff = torch.sum(torch.abs(actions[1:] - actions[:-1]), dim=-1)
    smoothness_temp = 0.5
    smoothness_penalty = -torch.exp(smoothness_temp * action_diff)
    
    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled + smoothness_penalty

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "smoothness_penalty": smoothness_penalty
    }

    return total_reward, reward_dict
