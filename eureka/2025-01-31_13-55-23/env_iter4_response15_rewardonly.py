@torch.jit.script
def compute_reward(root_states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract velocity and calculate forward velocity (positive x-axis)
    velocity = root_states[:, 7:10]
    forward_velocity = velocity[:, 0]

    # Adjusted forward velocity reward scaling
    forward_velocity_scale = 0.8  # Reduced to ensure balance with other components
    forward_velocity_temp = 0.1
    forward_velocity_reward = forward_velocity_scale * torch.exp(forward_velocity_temp * forward_velocity)

    # Enhanced energy penalty
    energy_penalty = torch.sum(actions**2, dim=-1)
    energy_temp = 1.5  # Increase temperature for noticeable penalty effect
    energy_penalty_scaled = -torch.exp(energy_temp * energy_penalty)

    # Boosted stable motion penalty
    stable_motion_penalty = torch.sum((actions[:, 1:] - actions[:, :-1])**2, dim=-1)
    stable_motion_temp = 1.0  # Increase to enhance penalty impact
    stable_motion_penalty_scaled = -torch.exp(stable_motion_temp * stable_motion_penalty)

    # Total reward combining all component with moderated transformation
    total_temp = 0.05
    total_reward = torch.exp(total_temp * (forward_velocity_reward + energy_penalty_scaled + stable_motion_penalty_scaled))

    # Reward components
    reward_dict = {
        "forward_velocity_reward": forward_velocity_reward,
        "energy_penalty_scaled": energy_penalty_scaled,
        "stable_motion_penalty_scaled": stable_motion_penalty_scaled
    }

    return total_reward, reward_dict
