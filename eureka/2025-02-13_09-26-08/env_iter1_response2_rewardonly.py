@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for reducing the distance to the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -distance_to_handle
    temperature_distance = 0.2  # Adjusted to increase difficulty
    transformed_dist_reward = torch.exp(dist_to_handle_reward / temperature_distance)

    # New approach to reward the opening of the door by scaling the joint position
    door_opening_reward = torch.clip(cabinet_dof_pos[:, 3], 0.0, 1.0)  # Ensuring it stays positive

    # New calculation for door velocity to ensure being positive means opening
    door_velocity_reward = torch.clip(cabinet_dof_vel[:, 3], 0.0, None)  # Only count opening as positive

    # Combine all components, re-weighted for better policy influence
    total_reward = 0.3 * transformed_dist_reward + 0.4 * door_opening_reward + 0.3 * door_velocity_reward

    # Return individual components for debugging and analysis
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward,
        "transformed_distance_reward": transformed_dist_reward
    }

    return total_reward, reward_components
