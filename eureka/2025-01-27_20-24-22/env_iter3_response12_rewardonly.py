@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted Distance Reward
    distance_reward_temperature = 0.1
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.5
    door_opening_reward = torch.tanh(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Refreshed Movement Penalty with New Scheme
    movement_penalty_temperature = 0.3
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # Redesigned and Amplified Task Completion Reward
    task_completion_threshold = 0.8
    task_completion = (cabinet_dof_pos[:, 3] > task_completion_threshold).float()
    task_completion_reward = task_completion * 5.0  # Provide a large reward incentive for task success

    # Total reward - Rebalancing Components
    total_reward = 1.0 * distance_reward + 0.5 * door_opening_reward - 0.3 * movement_penalty + 0.8 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
