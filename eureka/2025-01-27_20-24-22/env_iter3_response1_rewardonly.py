@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = 1.0 - torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.3
    door_opening_progress = torch.sigmoid(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Updated Movement Encouragement
    door_velocity = cabinet_dof_vel[:, 3]
    movement_bonus_temperature = 0.2
    movement_bonus = torch.exp(torch.abs(door_velocity) / movement_bonus_temperature) - 1.0

    # Revised Task Completion Incentive
    task_completion_threshold = 0.8  # Adjusted threshold for declaring task completion
    task_completion = (cabinet_dof_pos[:, 3] > task_completion_threshold).float()
    task_completion_reward = task_completion * 3.0  # High incentive for successful task completion

    # Total reward
    total_reward = (0.4 * distance_reward 
                    + 0.3 * door_opening_progress 
                    + 0.2 * movement_bonus 
                    + 0.1 * task_completion_reward)

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_progress': door_opening_progress,
        'movement_bonus': movement_bonus,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
