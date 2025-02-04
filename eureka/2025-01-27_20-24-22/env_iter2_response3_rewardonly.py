@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Re-scaled and Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    improved_distance_reward_temperature = 0.5
    improved_distance_reward = torch.exp(-distance_to_handle / improved_distance_reward_temperature)
    
    # Re-scaling Door Opening Reward
    normalized_door_opening_reward_temperature = 1.0
    door_opening_reward = cabinet_dof_pos[:, 3] / normalized_door_opening_reward_temperature

    # Revised Movement Penalty
    no_movement_threshold = 0.05  # Introduced threshold to classify active movement
    movement_penalty_temperature = 0.1
    movement_penalty = torch.where(torch.abs(cabinet_dof_vel[:, 3]) < no_movement_threshold,
                                   torch.exp(-1.0 / movement_penalty_temperature),
                                   torch.zeros_like(cabinet_dof_vel[:, 3]))

    # Encouragement for Successful Task Completion
    task_completion_bonus = 2.0
    task_completion_reward = torch.where(door_opening_reward > 1.0, 
                                         torch.tensor(task_completion_bonus, device=franka_grasp_pos.device), 
                                         torch.tensor(0.0, device=franka_grasp_pos.device))
    
    # Total reward calculation with balanced contributions
    total_reward = (0.4 * improved_distance_reward 
                    + 0.3 * door_opening_reward 
                    - 0.2 * movement_penalty 
                    + task_completion_reward)

    # Return total reward and component breakdown
    reward_dict = {
        'improved_distance_reward': improved_distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
