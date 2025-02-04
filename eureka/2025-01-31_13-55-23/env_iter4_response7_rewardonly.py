@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Enhanced forward velocity reward with emphasized scaling
    forward_velocity_scale = 2.5
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Re-evaluated energy penalty to allow more exploration
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.8  # Decrease to allow more exploration, reducing excessive penalization
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Reconsider stable motion penalty with increased contribution
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 0.7  # Tuning for potential improvement
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Total reward with enhanced overall temp scaling for component consolidation
    overall_temp = 0.15  # Improved wider exploration of better policies
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
