@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Normalized Door Opening Reward
    normalized_dof_pos = (cabinet_dof_pos[:, 3] - 0.0) / (0.8 - 0.0)  # assuming 0.0 to 0.8 as the operational range
    door_opening_reward_temperature = 0.1
    door_opening_reward = torch.exp(normalized_dof_pos / door_opening_reward_temperature) - 1.0

    # Revised Movement Penalty
    movement_penalty_temperature = 1.0
    movement_penalty = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Enabled Task Completion Reward
    task_completion_condition = (cabinet_dof_pos[:, 3] > 0.7).float()
    task_completion_reward = task_completion_condition * 2.0

    # Total Reward with Rescaled Components
    total_reward = 0.4 * distance_reward + 0.3 * door_opening_reward - 0.2 * movement_penalty + 0.1 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
