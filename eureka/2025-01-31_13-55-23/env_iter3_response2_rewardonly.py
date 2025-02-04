@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Scale the forward velocity reward to balance magnitude 
    forward_velocity_scale = 1.0  # Slightly scaled down for better balance
    forward_velocity_reward = forward_velocity_scale * forward_velocity
    
    # Energy penalty modification
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Adjust energy penalty temperature for better optimization
    energy_temp = 0.5  # Reduced to allow more influence of the penalty
    energy_penalty_scaled = -energy_temp * energy_penalty  
    
    # Overall temperature applied to sum of various components
    overall_temp = 0.1
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
