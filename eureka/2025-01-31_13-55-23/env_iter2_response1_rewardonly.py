@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis

    # Enhanced reward for forward velocity
    velocity_scale = 2.0  # Increased scale for stronger encouragement
    forward_velocity_reward = velocity_scale * forward_velocity
    
    # Stronger energy penalty linear transformation
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0  # Increased temperature for stronger penalization
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
