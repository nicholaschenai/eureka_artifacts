@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocities from root states
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Key reward component: forward velocity
    forward_velocity_scale = 1.8
    forward_velocity_reward = forward_velocity_scale * forward_velocity

    # Revised energy penalty with larger impact
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 2.0  # Enhance the temperature's impact
    energy_penalty_scaled = -energy_temp * energy_penalty

    # Update stable motion penalty with enhanced penalty condition
    stable_motion_penalty = torch.sum(torch.abs(actions[:, 1:] - actions[:, :-1]), dim=-1)
    stable_motion_temp = 1.0  # Increase to ensure it influences effectively
    stable_motion_penalty_scaled = -stable_motion_temp * stable_motion_penalty

    # Synthesize the total reward with appropriate temperature scaling
    # Normalization should ensure each component contributes
    overall_temp = 0.2
    total_reward = torch.exp(overall_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Break down components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
