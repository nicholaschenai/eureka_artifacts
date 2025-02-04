@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward with adjusted scale
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)
    
    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.25  # Increased temperature to reduce influence
    scaled_door_opening = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0
    
    # Adjusted Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.15  # Keeping similar scale for consistency
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)
    
    # Revised Task Completion Reward
    completion_threshold = 0.5
    task_completion_condition = (cabinet_dof_pos[:, 3] > completion_threshold)
    task_completion_reward = task_completion_condition.float() * 5.0  # Increased reward for task success
    
    # Total reward with adjusted weights
    total_reward = (
        0.5 * distance_reward +
        0.2 * scaled_door_opening -
        0.1 * movement_penalty +
        0.2 * task_completion_reward
    )
    
    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': scaled_door_opening,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
