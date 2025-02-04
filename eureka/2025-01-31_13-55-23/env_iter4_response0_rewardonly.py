@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjust forward velocity reward scaling
    forward_velocity_scale = 1.0  # Reduce to balance with other components
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Revised energy penalty with enhanced impact
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 2.0  # Increased for stronger discouragement of energy waste
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Enhanced stable motion penalty
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0  # Increased for more significant influence
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Total reward
    overall_temp = 0.1  # Moderate temp for normalizing total reward
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
