@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.1
    distance_reward = 1.0 - torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled and Modified Door Opening Reward
    door_opening_decrement = cabinet_dof_pos[:, 3] * torch.clamp_max(cabinet_dof_pos[:, 3], 1.0)
    door_opening_reward = door_opening_decrement * 3.0

    # Adaptive Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.2
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # Task Completion Reward at 70% open
    task_completion = (cabinet_dof_pos[:, 3] > 0.7).float() * 5.0

    # Total reward composition
    total_reward = 0.15 * distance_reward + 0.4 * door_opening_reward - 0.3 * movement_penalty + 0.45 * task_completion

    # Reward component dictionary
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion
    }

    return total_reward, reward_dict
