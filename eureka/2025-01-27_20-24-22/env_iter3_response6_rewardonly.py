@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced and Scaled Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5  # More sensitive scaling
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Regulated Door Opening Reward
    door_opening_factor = cabinet_dof_pos[:, 3]
    door_opening_reward_temperature = 0.2  # Reduced scaling
    door_opening_reward = torch.exp(door_opening_factor / door_opening_reward_temperature) - 1.0

    # Dynamic Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.1
    movement_penalty = torch.exp(-1.0 * torch.abs(door_velocity) / movement_penalty_temperature)

    # Progressive Task Completion Reward
    task_completion_reward = torch.clamp(door_opening_factor / 0.5, max=1.0) * 3.0  # Reward for reaching mid-point, incentivizes progressive achievement

    # Combining all components
    total_reward = 0.4 * distance_reward + 0.2 * door_opening_reward - 0.2 * movement_penalty + 0.2 * task_completion_reward

    # Individual reward components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    
    return total_reward, reward_dict
