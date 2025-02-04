@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis direction)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Enhanced forward velocity reward
    forward_velocity_scale = 3.0  # Increased scale factor to encourage speed
    forward_velocity_temp = 0.5
    forward_velocity_reward = forward_velocity_scale * forward_velocity
    forward_velocity_reward = torch.exp(forward_velocity_temp * forward_velocity_reward)

    # Revised energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 2.0  # Increased temperature for greater impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Revised stable motion penalty for more influence
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Total reward with improved balancing
    total_temp = 0.1
    total_reward = torch.exp(total_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
