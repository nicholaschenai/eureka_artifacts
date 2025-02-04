@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward with improved scale
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Re-scaled Door Opening Reward
    door_opening_progress = cabinet_dof_pos[:, 3]
    door_opening_reward_temperature = 20.0
    door_opening_reward = (torch.exp(door_opening_progress / door_opening_reward_temperature) - 1.0) / 100.0

    # Discard Movement Penalty (assuming no longer deemed valuable)

    # Redefine Task Completion Reward with more achievable criteria
    completion_threshold = 0.6  # Threshold adjusted for task completion
    task_completion_reward = (door_opening_progress > completion_threshold).float() * 10.0

    # Total Reward: Emphasis on task completion while maintaining balance
    total_reward = 0.3 * distance_reward + 0.4 * door_opening_reward + 0.3 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
