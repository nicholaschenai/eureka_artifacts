@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6  # Avoid division by zero
    temperature_distance = 0.1  # Further reduced for sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 5.0  # Increase scale for importance

    # Rescaled Door Opening Incentive
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_opening = 0.5  # Adjusted for moderation
    opening_restored_reward = torch.exp(-torch.abs(door_open_value - 1.0) / temperature_opening) * 2.0

    # An alternative approach for Velocity is assessing consistency in opening by speed
    speed_factor = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.5  # Ensure variation can be detected
    velocity_reward = torch.exp(-torch.abs(speed_factor) / temperature_velocity) * 0.1  # Reducing influence

    # Encourage shorter paths by promoting smooth, concise movements
    smoothness_measure = torch.mean(torch.abs(cabinet_dof_vel[:, 3] - door_open_value) + 1e-6)
    temperature_smoothness = 0.3
    smoothness_reward = torch.exp(-smoothness_measure / temperature_smoothness) * 0.5  # Introduce balance for fluidity

    # Aggregate total reward
    total_reward = dist_reward + opening_restored_reward + velocity_reward + smoothness_reward

    # Provide component-wise rewards for analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward,
        "smoothness_reward": smoothness_reward
    }

    return total_reward, reward_components
