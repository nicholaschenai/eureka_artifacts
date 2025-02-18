@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Adjusted for more sensitivity in differentiation
    transformed_dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Boosting reward for opening the cabinet door
    # Increased weight to emphasize its importance
    door_opening_reward = 2.0 * cabinet_dof_pos[:, 3]  # Double the previous influence 

    # Velocity: important factor. Scale it up and encourage positive velocity
    door_velocity_reward = 1.0 * torch.clamp(cabinet_dof_vel[:, 3], min=0.0)  # Encourage positive direction

    # Combine all reward components
    total_reward = (
        1.0 * transformed_dist_reward +
        1.5 * door_opening_reward +
        1.0 * door_velocity_reward
    )

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": distance_to_handle,
        "transformed_distance_reward": transformed_dist_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward
    }

    return total_reward, reward_components
