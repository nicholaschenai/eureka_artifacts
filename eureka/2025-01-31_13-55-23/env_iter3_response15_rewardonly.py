@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]  # Forward direction is along the x-axis

    # Scale the forward velocity reward
    forward_velocity_scale = 1.0  # Reduced to balance with energy penalty
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Energy penalty component re-worked
    energy_penalty_scale = 2.0  # Increased to provide more emphasis
    energy_penalty = energy_penalty_scale * torch.sum(actions**2, dim=-1)

    # Introduce a small penalty for deviation from a stable posture
    posture_penalty_scale = 0.1
    posture_penalty = posture_penalty_scale * torch.abs(root_states[:, 2] - 0.5)  # Assume an ideal height of 0.5

    # Calculate total reward
    overall_temp = 0.1  # For better balancing and stabilization
    total_reward = torch.exp(overall_temp * (forward_velocity_reward - energy_penalty - posture_penalty))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": -energy_penalty,
        "posture_penalty": -posture_penalty
    }

    return total_reward, reward_dict
