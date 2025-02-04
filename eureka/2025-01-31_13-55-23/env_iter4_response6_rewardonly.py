@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Enhanced forward velocity reward
    forward_velocity_scale = 3.0  # Further increase to emphasize forward velocity
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Adjust energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.5  # Adjusted for a stronger penalty
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Stable motion penalty revision
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0  # Increased to discourage abrupt changes
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Total reward combining all components with an enhanced moderate temp scaling
    overall_temp = 0.1
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
