@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Forward velocity reward with normalization
    forward_velocity_temp = 0.5  # Temperature for forward velocity
    forward_velocity_reward = torch.exp(forward_velocity_temp * forward_velocity)

    # Energy penalty - emphasize inefficiency
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_penalty_temp = 2.0  # Increased temperature for stronger penalty impact
    energy_penalty_scaled = -torch.exp(energy_penalty_temp * energy_penalty)

    # Revised stable motion penalty
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0
    stable_motion_penalty_scaled = -torch.exp(stable_motion_temp * stable_motion_penalty)

    # Overall reward combining all components with balanced impact
    total_reward = forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled

    # Reward components for monitoring
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
