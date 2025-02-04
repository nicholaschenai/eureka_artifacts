@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Enhanced forward velocity reward
    forward_velocity_scale = 2.0  # Slight increase to emphasize forward motion
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Revised energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.2  # Temperature adjustment for stronger impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Consider an additional stability or efficiency factor (e.g., reduce abrupt actions)
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 0.5
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Total reward combining all component with a moderate temp scaling
    overall_temp = 0.1
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
