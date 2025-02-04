@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 20.0  # Significantly increased scale
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0

    # Enhanced Movement Penalty
    velocity_condition = (torch.abs(cabinet_dof_vel[:, 3]) > 0.01).float()
    movement_penalty_temperature = 0.10
    movement_penalty = velocity_condition * torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Updated Task Completion Reward
    completion_condition = (cabinet_dof_pos[:, 3] > 0.7).float()  # Reflecting task achievement requirement
    task_completion_reward = completion_condition * 10.0  # Encouraging successful task completion

    # Total Reward Calculation
    total_reward = 0.4 * distance_reward + 0.4 * door_opening_reward - 0.1 * movement_penalty + 0.1 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
