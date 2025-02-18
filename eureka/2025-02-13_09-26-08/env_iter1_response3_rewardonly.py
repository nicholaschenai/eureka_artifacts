@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.5  # Adjusted temperature for a moderate transformation
    dist_to_handle_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Reward for the drawer position indicating it is opening or fully open
    door_target_position = torch.full_like(cabinet_dof_pos[:, 3], 1.0)  # Assume 1.0 is fully open
    door_opening_reward = torch.clamp(cabinet_dof_pos[:, 3], min=0.0)
    door_opening_complete_reward = torch.abs(door_target_position - cabinet_dof_pos[:, 3]) <= 0.1  # Threshold for success
    transformed_door_opening = torch.where(door_opening_complete_reward, torch.tensor(1.0, device=cabinet_dof_pos.device), 0.1 * door_opening_reward)

    # Reward for the speed in which the door is being opened in the correct direction
    door_velocity_reward = torch.where(cabinet_dof_vel[:, 3] > 0.0, cabinet_dof_vel[:, 3], torch.tensor(0.0, device=cabinet_dof_vel.device))

    # Combine all reward components with adjusted weights
    total_reward = 0.2 * dist_to_handle_reward + 1.0 * transformed_door_opening + 0.1 * door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "transformed_door_opening": transformed_door_opening,
        "door_velocity": door_velocity_reward
    }

    return total_reward, reward_components
