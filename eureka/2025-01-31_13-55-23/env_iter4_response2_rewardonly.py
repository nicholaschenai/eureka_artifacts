@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocities
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Forward velocity reward with enhanced importance
    forward_velocity_scale = 2.5
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Improved energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.5  # Increased sensitivity
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Enhanced stable motion penalty to promote smoother movements
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 0.8  # Strengthened temperature for visible impact
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Combine components using a moderate temperature scaling
    overall_temp = 0.1
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
