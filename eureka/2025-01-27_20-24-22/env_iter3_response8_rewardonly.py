@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.35
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Revised Movement Penalty
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    movement_penalty = torch.clamp(1.0 - door_velocity, min=0.0)

    # Enhanced Task Completion Reward
    target_position = 0.5  # New threshold for task completion
    task_completion = (cabinet_dof_pos[:, 3] >= target_position).float()
    task_completion_reward = task_completion * 3.0  # Enhanced reward for opening substantially

    # Total reward
    total_reward = 0.4 * distance_reward + 0.2 * door_opening_reward - 0.1 * movement_penalty + 0.3 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
