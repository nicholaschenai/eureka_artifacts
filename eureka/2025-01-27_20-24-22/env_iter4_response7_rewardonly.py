@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor,
                   drawer_grasp_pos: torch.Tensor,
                   cabinet_dof_pos: torch.Tensor,
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                   
    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5  # Increased temperature for greater influence
    distance_reward = 1.0 - torch.tanh(distance_to_handle / distance_reward_temperature)

    # Scaled Door Opening Reward
    door_opening_reward_temperature = 500.0  # Temperaturé adjusted to reduce scale
    door_opening_reward = torch.tanh(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Redesigned Movement Penalty
    velocity_penalty_multiplier = 0.5
    movement_penalty = velocity_penalty_multiplier * torch.mean(cabinet_dof_vel[:, 3].abs(), dim=-1)

    # Structured Task Completion Reward
    completion_threshold = 0.8  # Slightly increased threshold for completion
    task_completion_reward = (cabinet_dof_pos[:, 3] > completion_threshold).float() * 5.0

    # Total reward
    total_reward = (0.4 * distance_reward + 
                    0.3 * door_opening_reward - 
                    0.2 * movement_penalty + 
                    0.3 * task_completion_reward)

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'task_completion_reward': task_completion_reward
    }
    return total_reward, reward_dict
