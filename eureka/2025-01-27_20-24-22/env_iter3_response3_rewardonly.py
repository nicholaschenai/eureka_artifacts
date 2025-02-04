@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.3
    scaled_door_opening = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0

    # Rewritten Movement Penalty
    velocity_threshold = 0.05  # Allow minimal velocity
    excessive_movement_penalty_temperature = 0.2
    movement_penalty = torch.where(
        torch.abs(cabinet_dof_vel[:, 3]) < velocity_threshold,
        torch.tensor(0.0, device=franka_grasp_pos.device),
        torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / excessive_movement_penalty_temperature)
    )

    # Rewritten Task Completion Reward
    task_completion_threshold = 0.8  # A more reachable threshold
    task_completion = (cabinet_dof_pos[:, 3] > task_completion_threshold).float()
    task_completion_reward = task_completion * 5.0  # Strong incentive for success

    # Total reward
    total_reward = 0.4 * distance_reward + 0.3 * scaled_door_opening - 0.2 * movement_penalty + 0.5 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': scaled_door_opening,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
