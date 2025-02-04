@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2
    distance_reward = 1.0 - torch.exp(-distance_to_handle / distance_reward_temperature)

    # Re-scaled Door Opening Reward
    door_opening_target = 1.0
    door_opening_reward_temperature = 1000.0
    door_opening_reward = (torch.tanh(cabinet_dof_pos[:, 3]) / door_opening_target) * door_opening_reward_temperature

    # Revised Movement Penalty
    movement_penalty_temperature = 1.0
    movement_penalty = -torch.exp(torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Enhanced Task Completion Reward
    task_completion_condition = (cabinet_dof_pos[:, 3] > 0.95).float()
    task_completion_reward = task_completion_condition * 10.0

    # Calculate Total Reward
    total_reward = 0.5 * distance_reward + 0.2 * door_opening_reward + 0.3 * task_completion_reward + movement_penalty

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
