@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Forward velocity reward (slightly adjusted scale)
    forward_velocity_scale = 1.5  # Balanced to prevent overshadowing
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Adjusted energy penalty with reduced temperature
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 0.8  # Further reduced temperature to highlight penalty
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Revised stable motion penalty with increased scale
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_scale = 0.6  # Increase scale to improve discouragement
    stable_motion_penalty_scaled = -stable_motion_scale * stable_motion_penalty

    # Normalize the total reward
    overall_temp = 0.1  # Adjusted to ensure the components are balanced
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
