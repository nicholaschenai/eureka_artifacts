@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]
    
    # Adjusted forward velocity reward
    forward_velocity_scale = 2.5  # Slightly increase to push for higher speeds
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Refined energy penalty: still penalizing energy use but trying to encourage efficiency
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.0  # Slightly adjusted to make it influential but not dominant
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Stable motion penalty: prioritize smoother, stable movement
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0  # Adjusted to ensure this contributes significantly
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Total reward: Having a dynamic temperature or scale can help agent explore aspects uniformly
    overall_temp = 0.2
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components dictionary
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
