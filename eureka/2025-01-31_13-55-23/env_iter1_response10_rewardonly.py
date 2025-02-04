@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, torso_rotation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    velocity = root_states[:, 7:10]  # Extract linear velocity from root states
    forward_velocity = velocity[:, 0]  # Assuming forward direction is along the x-axis
    
    # Reward for forward velocity
    forward_velocity_reward = forward_velocity
    
    # Adjust energy penalty scaling
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.5  # Increased temperature for stronger penalty
    energy_penalty_scaled = torch.exp(-energy_temp * energy_penalty)
    
    # Stability reward to maintain upright orientation
    # Assuming upright direction is the z-axis (index 2)
    upright_penalty = torch.abs(torso_rotation[:, 2])  # Penalize deviation from upright
    upright_temp = 1.0
    upright_penalty_scaled = torch.exp(-upright_temp * upright_penalty)
    
    # Total reward
    total_reward = forward_velocity_reward + energy_penalty_scaled + upright_penalty_scaled
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "upright_penalty_scaled": upright_penalty_scaled
    }

    return total_reward, reward_dict
