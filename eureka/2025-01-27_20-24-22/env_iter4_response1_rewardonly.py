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
    door_opening_coef = 0.1
    door_opening_reward = door_opening_coef * cabinet_dof_pos[:, 3]

    # Enhanced Movement Penalty
    movement_penalty = -torch.abs(cabinet_dof_vel[:, 3])

    # Reworked Task Completion Reward
    completion_threshold = 0.8
    task_completion_bonus = 10.0
    task_completion_reward = (cabinet_dof_pos[:, 3] >= completion_threshold).float() * task_completion_bonus

    # Total reward
    total_reward = distance_reward + door_opening_reward + movement_penalty + task_completion_reward

    # Return total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
