@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Adjusted sensitivity
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)
    
    # Improved Door Opening Reward
    door_opening_reward_temperature = 0.5  # Scaling down
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)
    
    # Enhanced Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.5  # Adjusted to enhance effectiveness
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # Task Completion Bonus
    door_opened_threshold = 0.9  # Assume a threshold for door being significantly open
    task_completion_bonus = torch.where(cabinet_dof_pos[:, 3] > door_opened_threshold, 
                                        torch.tensor(1.0, device=franka_grasp_pos.device), 
                                        torch.tensor(0.0, device=franka_grasp_pos.device))

    # Total reward calculation with normalized contributions
    total_reward = 0.3 * distance_reward + 0.3 * door_opening_reward - 0.2 * movement_penalty + 0.3 * task_completion_bonus

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_bonus': task_completion_bonus
    }
    return total_reward, reward_dict
