@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced and Rescaled Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.05
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Re-scaled Door Opening Reward
    door_opening_target = 1.0  # Target cabinet opening position
    door_opening_error = torch.clamp(door_opening_target - cabinet_dof_pos[:, 3], min=0.0)
    door_opening_reward_temperature = 0.01
    door_opening_reward = torch.exp(-door_opening_error / door_opening_reward_temperature)

    # Refined Movement Penalty
    excess_velocity = torch.clip(torch.abs(cabinet_dof_vel[:, 3]) - 0.01, min=0.0)
    movement_penalty_temperature = 0.01
    movement_penalty = torch.exp(-excess_velocity / movement_penalty_temperature)

    # Effective Task Completion Reward
    completion_threshold = 0.8  # More achievable completion threshold
    task_completion_condition = (cabinet_dof_pos[:, 3] >= completion_threshold).float()
    task_completion_reward = task_completion_condition * 10.0

    # Calculate Total Reward
    total_reward = 0.25 * distance_reward + 0.25 * door_opening_reward - 0.1 * movement_penalty + 0.4 * task_completion_reward

    # Return the total reward and individual reward components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
