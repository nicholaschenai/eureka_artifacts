@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusted forward velocity reward
    forward_velocity_scale = 2.5  # Further emphasized
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Adjusted energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.4  # Strengthened for better impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Adjusted stability penalty to ensure smoother motion
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 0.8  # Moderate scale
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Overall reward calculation combining all components
    overall_temp = 0.15  # Slightly increased temperature for balancing reward components
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
