@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_to_handle_reward = -distance_to_handle

    # Reward component for opening the cabinet door (measured by the joint position of the door)
    door_opening_reward = cabinet_dof_pos[:, 3]  # Assuming index 3 corresponds to the door position

    # Reward component for velocity of the door opening (positive for opening direction)
    door_velocity_reward = cabinet_dof_vel[:, 3]  # Assuming index 3 corresponds to the door velocity

    # Transform the distance reward using a temperature parameter
    temperature_distance = 0.05  # Adjusted for more moderate scaling
    transformed_dist_reward = torch.exp(dist_to_handle_reward / temperature_distance)

    # Reward for achieving the task (e.g., door fully opened)
    task_success_threshold = 0.5  # Arbitrary threshold for door opening
    task_success_reward = torch.where(cabinet_dof_pos[:, 3] > task_success_threshold, torch.tensor(1.0, device=cabinet_dof_pos.device), torch.tensor(0.0, device=cabinet_dof_pos.device))

    # Scale up the door opening and velocity rewards
    scaled_door_opening_reward = 3.0 * door_opening_reward  # Scaling factor
    scaled_door_velocity_reward = 2.0 * door_velocity_reward  # Scaling factor

    # Combine all reward components
    total_reward = 0.3 * transformed_dist_reward + scaled_door_opening_reward + 0.5 * scaled_door_velocity_reward + 2.0 * task_success_reward

    # Collect the individual components in a dictionary
    reward_components = {
        "distance_to_handle": dist_to_handle_reward,
        "door_opening": scaled_door_opening_reward,
        "door_velocity": scaled_door_velocity_reward,
        "transformed_distance_reward": transformed_dist_reward,
        "task_success": task_success_reward
    }

    return total_reward, reward_components
