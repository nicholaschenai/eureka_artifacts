@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -0.1 * distance_to_handle  # Slightly rescaled

    # Reward component for opening the cabinet door (measured by the joint position of the door)
    door_opening_reward = 1.0 * cabinet_dof_pos[:, 3]  # Higher coefficient for more emphasis

    # Reward component for velocity of the door opening (positive for opening direction)
    door_velocity_reward = 0.5 * torch.abs(cabinet_dof_vel[:, 3])  # Consider absolute value for positive change

    # Transform the distance reward using a temperature parameter
    temperature_distance = 0.05  # More aggressively encouraging closer proximity
    transformed_dist_reward = torch.exp(dist_to_handle_reward / temperature_distance)

    # Combine all reward components
    total_reward = 0.3 * transformed_dist_reward + 1.5 * door_opening_reward + 0.7 * door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward,
        "transformed_distance_reward": transformed_dist_reward
    }

    return total_reward, reward_components
