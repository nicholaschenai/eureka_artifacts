@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Rescaled Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Adjusted for more prominence
    distance_reward = 1.0 - torch.tanh(distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 500.0  # Significantly increased to reduce influence
    door_opening_reward = cabinet_dof_pos[:, 3] / door_opening_reward_temperature

    # Refined Movement Penalty
    movement_penalty_temperature = 0.1  # Increased for noticeable effect
    movement_penalty = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Revamped Task Completion Reward
    completion_threshold = 0.7  # Threshold for door considered open
    task_completion_reward = (cabinet_dof_pos[:, 3] >= completion_threshold).float() * 10.0  # Substantial bonus to emphasize completion

    # Total reward
    total_reward = 0.25 * distance_reward + 0.25 * door_opening_reward - 0.25 * movement_penalty + 0.25 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
