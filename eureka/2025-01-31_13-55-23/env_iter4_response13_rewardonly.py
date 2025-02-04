@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Forward velocity reward
    forward_velocity_scale = 2.5  # Increased scale for greater impact
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Adjusted energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.5  # Increased temperature for stronger impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # More significant stable motion penalty
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0  # Increased temperature for better jitter control
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Bonus for running efficiently in a stable manner
    stability_bonus = torch.exp(-torch.abs(velocity[:, 1]) - torch.abs(velocity[:, 2]))  # Encouraging stability in y and z axes
    stability_bonus_temp = 0.3
    stability_bonus_scaled = stability_bonus_temp * stability_bonus

    # Total reward, balanced with a moderate overall temp scaling
    overall_temp = 0.2
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled + stability_bonus_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled,
        "stability_bonus_scaled": stability_bonus_scaled
    }

    return total_reward, reward_dict
