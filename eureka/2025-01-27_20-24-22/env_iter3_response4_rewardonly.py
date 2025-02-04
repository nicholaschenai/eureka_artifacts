@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.1  # More sensitive temperature for better distinction
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.5  # Rescale to control magnitude
    scaled_door_opening = torch.tanh(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Adjusted Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.2  # Slight increase in penalty sensitivity
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # Revised Task Completion Incentive
    task_completion_threshold = 0.5
    task_completion = (cabinet_dof_pos[:, 3] >= task_completion_threshold).float()
    task_completion_reward = task_completion * 5.0  # Increased incentive for task completion

    # Total reward
    total_reward = 0.2 * distance_reward + 0.4 * scaled_door_opening - 0.3 * movement_penalty + 0.1 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': scaled_door_opening,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
