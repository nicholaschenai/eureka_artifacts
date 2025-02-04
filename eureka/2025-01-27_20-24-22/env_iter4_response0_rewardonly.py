@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.15  # Increased temperature for smoother reward gradient
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.005  # Significantly reduced temperature for better scaling
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0

    # Revised Movement Penalty
    movement_penalty_temperature = 0.01 
    movement_penalty = torch.sum(torch.abs(cabinet_dof_vel), dim=-1) * torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # Revised Task Completion Reward
    completion_threshold = 0.5  # Lowered the threshold for achievability
    completion_condition = (cabinet_dof_pos[:, 3] > completion_threshold).float()
    task_completion_reward = completion_condition * 5.0  # Maintain a strong incentive on completion

    # Total reward
    total_reward = 0.4 * distance_reward + 0.1 * door_opening_reward - 0.1 * movement_penalty + 0.4 * task_completion_reward

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
