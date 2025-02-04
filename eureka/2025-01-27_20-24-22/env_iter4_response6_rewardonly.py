@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced and Scaled Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.1
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature) * 2.0

    # Scaled Down Door Opening Reward
    door_opening_reward_temperature = 15.0
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0

    # More Prominent Movement Penalty
    velocity_threshold = 0.015
    movement_penalty = (torch.abs(cabinet_dof_vel[:, 3]) > velocity_threshold).float()
    movement_penalty *= -0.1  # Reduced penalty but present to discourage excess movement

    # Reintroduced Task Completion Reward
    completion_condition = (cabinet_dof_pos[:, 3] > 0.5).float()  # Reduced threshold for realism
    task_completion_reward = completion_condition * 2.0  # Apply significant bonus for task success

    # Total reward combining balanced component contributions
    total_reward = 0.4 * distance_reward + 0.2 * door_opening_reward + movement_penalty + 0.4 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
