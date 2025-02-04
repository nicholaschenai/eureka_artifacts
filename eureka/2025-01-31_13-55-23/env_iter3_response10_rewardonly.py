@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Forward velocity reward - adjusted scaling
    forward_velocity_scale = 1.0  # Slightly reduced for better balance
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Improved energy penalty calculation
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Adjust energy penalty scaling for stronger penalization
    energy_temp = 1.5  # Increased from previous
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)

    # Introduce a reward transformation for better stabilization and balancing
    reward_transform_temp = 0.05
    total_reward = torch.exp(reward_transform_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
