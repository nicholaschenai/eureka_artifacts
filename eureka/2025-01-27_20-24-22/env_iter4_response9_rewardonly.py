@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                   
    # Enhanced Distance Reward for Effective Closeness Encouragement
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.05
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Controlled Door Opening Reward with Balanced Scaling
    door_opening_reward_temperature = 100.0  # Adjusted for range control
    door_opening_progress = torch.clamp(cabinet_dof_pos[:, 3], 0.0, 1.0)
    door_opening_reward = door_opening_progress / door_opening_reward_temperature

    # Redesigned Movement Penalty for Disincentivizing Unnecessary Dynamics
    movement_penalty = torch.abs(cabinet_dof_vel[:, 3])
    movement_penalty_temperature = 0.1
    scaled_movement_penalty = torch.exp(-movement_penalty / movement_penalty_temperature)

    # Reintroduced Task Completion Incentive for Encouraging Final Goal
    completion_condition = (cabinet_dof_pos[:, 3] > 0.5).float()  # Adjusted to a realistic threshold
    task_completion_reward = completion_condition * 5.0

    # Total reward computation with balanced weighting
    total_reward = 0.4 * distance_reward + 0.3 * door_opening_reward - 0.2 * scaled_movement_penalty + 0.5 * task_completion_reward

    # Return the total reward and individual components
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': scaled_movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
