@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temp = 0.2
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temp)

    # Rescaled Door Opening Reward
    door_opening_reward_temp = 0.1
    door_opening_reward = (torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temp) - 1.0) * 0.1

    # Adjusted Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temp = 0.2
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temp)

    # Reformed Task Completion Reward
    threshold = 0.5
    task_completion_reward = (cabinet_dof_pos[:, 3] > threshold).float() * 1.5  # higher incentive
    
    # Total reward recalibration
    total_reward = 0.4 * distance_reward + 0.4 * door_opening_reward - 0.1 * movement_penalty + 0.1 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
