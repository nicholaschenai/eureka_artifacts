@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Revised Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Increased sensitivity
    scaled_distance = torch.exp(-distance_to_handle / distance_reward_temperature)
    distance_reward = 1.5 * scaled_distance  # Increased influence

    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.5  # Reduce influence
    door_opening_reward = torch.tanh(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Enhanced Movement Penalty
    movement_penalty_temperature = 0.3
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # New Completion Bonus
    completion_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.5, torch.tensor(5.0, device=franka_grasp_pos.device), torch.tensor(0.0, device=franka_grasp_pos.device))

    # Total reward calculation
    total_reward = 0.6 * distance_reward + 0.3 * door_opening_reward - 0.2 * movement_penalty + 0.3 * completion_bonus

    # Components dictionary
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'completion_bonus': completion_bonus
    }
    return total_reward, reward_dict
