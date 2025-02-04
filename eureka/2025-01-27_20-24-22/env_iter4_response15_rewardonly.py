@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Increased Weight for Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.25
    distance_reward = 1.0 - torch.tanh(distance_to_handle / distance_reward_temperature)
    
    # Rescaled Door Opening Reward
    door_opening_reward_scale = 0.001
    door_opening_reward = door_opening_reward_scale * cabinet_dof_pos[:, 3]  # Adjust scaling

    # Reevaluated Movement Penalty
    movement_penalty = -0.01 * torch.abs(cabinet_dof_vel[:, 3])

    # Redesigned Task Completion Reward
    completion_threshold = 0.7
    completion_reward_value = 10.0
    task_completion_reward = torch.where(cabinet_dof_pos[:, 3] > completion_threshold, 
                                         torch.tensor(completion_reward_value, device=franka_grasp_pos.device), 
                                         torch.tensor(0.0, device=franka_grasp_pos.device))
    
    # Aggregate total reward
    total_reward = 0.3 * distance_reward + 0.2 * door_opening_reward + movement_penalty + task_completion_reward

    # Return the total reward and detailed component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
