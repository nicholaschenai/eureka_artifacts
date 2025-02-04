@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjusted Distance Reward: Increased sensitivity to distances
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.1
    distance_reward = 1.0 - torch.tanh(distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.3
    scaled_door_opening = torch.sigmoid(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Adjusted Movement Penalty: Steeper penalty for lack of movement
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.1
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # New Task Completion Reward: Reduced threshold for positive reinforcement
    task_completion_threshold = 0.2
    task_completion = (cabinet_dof_pos[:, 3] > task_completion_threshold).float() 
    task_completion_reward = task_completion * 5.0  # Strong incentive for achieving the task

    # Total reward with revised scaling
    total_reward = 0.2 * distance_reward + 0.4 * scaled_door_opening - 0.1 * movement_penalty + 0.3 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': scaled_door_opening,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
