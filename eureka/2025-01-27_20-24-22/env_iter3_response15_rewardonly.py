@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Properly Balanced Door Opening Reward
    door_opening_reward_temp = 5.0
    normalized_door_opening = cabinet_dof_pos[:, 3] / door_opening_reward_temp

    # Enhanced Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.1
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # Adjusted Task Completion Reward
    door_fully_open = (cabinet_dof_pos[:, 3] > 0.5).float()
    incrementally_open_reward = torch.where(cabinet_dof_pos[:, 3] > 0.1, torch.tensor(1.0).to(franka_grasp_pos.device), torch.tensor(0.0).to(franka_grasp_pos.device))

    # Total reward
    total_reward = (
        0.4 * distance_reward + 
        0.4 * normalized_door_opening - 
        0.2 * movement_penalty + 
        0.5 * door_fully_open + 
        0.1 * incrementally_open_reward
    )

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': normalized_door_opening,
        'movement_penalty': movement_penalty,
        'task_completion_reward': door_fully_open + incrementally_open_reward
    }
    return total_reward, reward_dict
