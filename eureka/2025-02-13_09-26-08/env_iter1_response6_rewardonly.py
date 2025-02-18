@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Updated reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.2  # Adjusted for more moderate transformation
    transformed_dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # New reward component for pulling the drawer open based on the change in position
    min_door_opening_pos = 0.1  # Threshold to start rewarding the door opening
    door_opening_reward = torch.where(cabinet_dof_pos[:, 3] > min_door_opening_pos, cabinet_dof_pos[:, 3], torch.zeros_like(cabinet_dof_pos[:, 3]))

    # Increasing weight on the velocity when it contributes to opening
    door_velocity_reward = torch.where(cabinet_dof_vel[:, 3] > 0, cabinet_dof_vel[:, 3], torch.zeros_like(cabinet_dof_vel[:, 3]))

    # Combine all reward components with adjusted weights to balance their influence
    total_reward = 1.0 * transformed_dist_reward + 2.0 * door_opening_reward + 1.5 * door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": transformed_dist_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward
    }

    return total_reward, reward_components
