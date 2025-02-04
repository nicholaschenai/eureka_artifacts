@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and compute forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusted forward velocity reward to maintain balance
    forward_velocity_scale = 1.5  # Reduce to lower the weight on forward velocity
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Revised energy penalty with increased temperature scaling
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 2.0  # Higher temp to increase penalty impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Modified stable motion penalty with increased temperature to promote smoother movements
    stable_motion_change = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0  # Higher temp to boost incentive for stable actions
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_change

    # Total reward applying temperature scaling to reduce bias
    overall_temp = 0.2
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
