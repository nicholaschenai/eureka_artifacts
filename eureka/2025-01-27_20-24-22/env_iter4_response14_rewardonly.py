@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_angle: torch.Tensor,
                   cabinet_velocity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Enhanced Distance Reward with reduced temperature
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    distance_reward_temperature = 0.03
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Controlled Door Opening Reward with reduced scale
    door_opening_reward_temperature = 0.02
    door_opening_reward = torch.exp(cabinet_angle[:, 3] / door_opening_reward_temperature) - 1.0

    # Revised Movement Penalty by reconsidering the temperature
    movement_penalty_temperature = 0.01
    movement_penalty = torch.exp(-torch.abs(cabinet_velocity[:, 3]) / movement_penalty_temperature)

    # New Task Completion Reward with relaxed conditions
    completion_threshold = 0.5  # Adjusted threshold for task completion
    task_completion_reward = (cabinet_angle[:, 3] > completion_threshold).float() * 3.0 

    # Total reward computation
    total_reward = 0.4 * distance_reward + 0.3 * door_opening_reward - 0.2 * movement_penalty + 0.5 * task_completion_reward

    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    
    return total_reward, reward_dict
