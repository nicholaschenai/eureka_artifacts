@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract the forward velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Re-scale the forward velocity reward to have a stronger positive effect
    forward_velocity_scale = 2.0
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Adjust the energy penalty by modifying the scaling factor
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Modify energy penalty scaling for a stronger negative effect
    energy_temp = 1.5
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Normalize the total reward to a manageable scale
    overall_temp = 0.1
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
