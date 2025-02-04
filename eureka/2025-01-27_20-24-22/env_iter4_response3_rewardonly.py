@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Enhanced Distance Reward to drive proximity to handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.05  # Smaller value for quicker rise
    distance_reward = 1.0 - torch.exp(-distance_to_handle / distance_reward_temperature)

    # Scaled Door Opening Reward for steady-built opening
    door_opening_temperature = 1000.0
    door_opening_reward = (torch.exp(cabinet_dof_pos[:, 3]) / door_opening_temperature) - 1.0

    # Redefined Movement Penalty to discourage noise
    excessive_movement_threshold = 0.01
    movement_condition = (torch.abs(cabinet_dof_vel[:, 3]) > excessive_movement_threshold).float()
    movement_penalty_temperature = 0.02
    movement_penalty = movement_condition * torch.exp(torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Reinvigorated Task Completion Incentive for full task success
    completion_condition = (cabinet_dof_pos[:, 3] > 0.8).float()  # Re-evaluated task completion
    task_completion_reward = completion_condition * 50.0  # Strong incentive for task finalization

    # Combined Reward Assembly
    total_reward = 0.4 * distance_reward + 0.2 * door_opening_reward - 0.3 * movement_penalty + 0.1 * task_completion_reward

    # Output reward composition
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
