@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Redefined reward for reducing distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.5  # Adjusted temperature for better scaling
    dist_reward = 1 / (1 + distance_to_handle)  # Inverse for better approach

    # Updated opening reward encouraging door opening immediately
    door_open_value = cabinet_dof_pos[:, 3]
    open_goal = 0.5  # Assume door is considered open at this value
    opening_restored_reward = 2.0 * (door_open_value >= open_goal).float()  # Provide a step function reward

    # Rescaled velocity reward to bring within range
    door_velocity = cabinet_dof_vel[:, 3].abs()
    velocity_scale = 0.1  # Adjusted scale
    velocity_reward = torch.clamp(door_velocity * velocity_scale, max=1.0)  # Ensure upper bound is capped

    # Rebalance overall total reward
    total_reward = 1.0 * dist_reward + 3.0 * opening_restored_reward + 0.5 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
