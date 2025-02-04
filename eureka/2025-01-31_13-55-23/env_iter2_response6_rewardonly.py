@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Enhancing the reward for forward velocity
    velocity_scale = 1.0  # Increased scale to give more weight to forward velocity
    forward_velocity_reward = velocity_scale * forward_velocity

    # Strengthening penalty for action energy usage
    energy_penalty = torch.sum(actions**2, dim=-1)
    
    # Further increasing the energy temperature for stronger penalization
    energy_temp = 1.0  # Increased from 0.5 to 1.0
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
