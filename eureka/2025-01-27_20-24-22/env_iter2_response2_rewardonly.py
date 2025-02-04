@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjusted Distance Reward - higher influence
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Higher sensitivity
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)
    
    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.5  # Reducing influence
    door_opening_reward = 2.0 * torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)
    
    # Reformulated Movement Penalty
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    movement_penalty_temperature = 0.1  # Increase sensitivity
    movement_penalty = torch.exp(-door_velocity / movement_penalty_temperature)
    
    # Additional Time Penalty to encourage faster completions
    time_penalty_temperature = 0.001
    time_penalty = torch.exp(-cabinet_dof_pos[:, 3] / time_penalty_temperature)
    
    # Total reward calculation with balanced weights and new components
    total_reward = 0.3 * distance_reward + 0.3 * door_opening_reward - 0.2 * movement_penalty - 0.2 * time_penalty
    
    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'time_penalty': time_penalty
    }
    
    return total_reward, reward_dict
