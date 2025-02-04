@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Scale the forward velocity reward to balance magnitude
    forward_velocity_scale = 1.2  # Adjusted down from previous scale
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Adjust energy penalty for more substantial variation
    energy_penalty = torch.sum(actions**2, dim=-1)

    # New energy penalty temperature to increase impact
    energy_temp = 2.0  # Increased temperature for more influence
    energy_penalty_scaled = -energy_temp * energy_penalty  # Enhanced linear scaling

    # Use adjusted temperature for the overall reward transformation
    overall_temp = 0.05
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled
    }

    return total_reward, reward_dict
