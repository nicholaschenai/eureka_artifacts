@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Adjust the forward velocity reward scale to balance its magnitude
    forward_velocity_scale = 1.0  # Reduced scale from prior versions
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Energy penalty adjustment for a stronger penalty impact
    energy_penalty = torch.sum(actions**2, dim=-1)

    # Adjust energy temperature to emphasize penalty
    energy_temp = 1.5  # Increased temperature for greater penalization impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Overall reward transformation
    overall_temp = 0.1
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))
    
    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
