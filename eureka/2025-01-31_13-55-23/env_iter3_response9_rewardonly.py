@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Scale the forward velocity reward 
    forward_velocity_scale = 2.0  # Increased to enhance impact
    forward_velocity_reward = forward_velocity_scale * forward_velocity
    
    # Compute energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Rescale and introduce variance in energy penalty
    energy_temp = 1.5  # Increased temperature to increase penalty influence
    energy_penalty_scaled = -energy_temp * energy_penalty  # Linearly scaled, emphasizing penalization

    # Adjusted exponential transformation for total reward normalization
    overall_temp = 0.02
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
