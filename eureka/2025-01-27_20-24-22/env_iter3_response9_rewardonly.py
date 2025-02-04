@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward with tighter temperature for more precision influence
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15
    distance_reward = 1.0 - torch.exp(-distance_to_handle / distance_reward_temperature)
    
    # Scaled Door Opening Reward
    door_opening_reward_temperature = 0.05
    scaled_door_opening = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0
    scaled_door_opening = torch.clamp(scaled_door_opening, max=1.0)
    
    # Refined Movement Penalty with adjusted scaling
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.05
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)
    movement_penalty = torch.clamp(movement_penalty, min=0.3)
    
    # Revised Task Completion Increment
    task_completion_increment = (cabinet_dof_pos[:, 3] > 0.3).float()
    task_completion_reward = task_completion_increment * (1.0 + cabinet_dof_pos[:, 3])
    
    # Total reward
    total_reward = 0.25 * distance_reward + 0.25 * scaled_door_opening - 0.1 * movement_penalty + 0.4 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': scaled_door_opening,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
