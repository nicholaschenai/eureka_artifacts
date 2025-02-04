@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract forward velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Adjust forward velocity reward to prevent dominance
    forward_velocity_scale = 0.5  # Reduced from 1.5 to reduce its dominance
    forward_velocity_reward = forward_velocity_scale * forward_velocity
    
    # Revise the energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Increased energy penalty scaling
    energy_temp = 2.5  # Increased to provide more penalization
    energy_penalty_scaled = -torch.exp(-energy_temp * energy_penalty)  # Exponential transformation for stronger nonlinear penalization
    
    # Combine components with appropriate normalization
    overall_temp = 0.1
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
