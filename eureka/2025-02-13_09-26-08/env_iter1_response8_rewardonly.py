@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -0.5 * distance_to_handle  # Reduce scale to make it significant but not overwhelming

    # Reward for opening the door (focus on increasing door opening angle)
    door_opening_threshold = 0.1  # Consider the door "opened" if moved more than this threshold
    door_opening_reward = torch.where(cabinet_dof_pos[:, 3] > door_opening_threshold, torch.ones_like(cabinet_dof_pos[:, 3]), torch.zeros_like(cabinet_dof_pos[:, 3]))

    # Reward based on velocity (encourage positive velocity for door opening)
    velocity_threshold = 0.01  # Velocity should be greater than threshold to be considered significant
    door_velocity_reward = torch.where(cabinet_dof_vel[:, 3] > velocity_threshold, 0.1 * cabinet_dof_vel[:, 3], torch.zeros_like(cabinet_dof_vel[:, 3]))

    # Assign the transformed distance reward a lower scale, focus more on door opening
    temperature_distance = 0.1  # Adjusted previously
    transformed_distance_reward = torch.exp(dist_to_handle_reward / temperature_distance)

    # Combine all reward components, emphasizing door opening
    total_reward = transformed_distance_reward + 2.0 * door_opening_reward + door_velocity_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "door_opening": door_opening_reward,
        "door_velocity": door_velocity_reward,
        "transformed_distance_reward": transformed_distance_reward
    }

    return total_reward, reward_components
