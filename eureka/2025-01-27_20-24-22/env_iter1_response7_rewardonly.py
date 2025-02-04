@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.3  # Increased sensitivity
    distance_reward = 1.0 / (1.0 + distance_to_handle / distance_reward_temperature)

    # Improved Door Opening Reward
    door_opening_reward_temperature = 0.3  # Adjusted to reduce influence
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Penalty for not changing the door state (encourages actual door movement)
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.2
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)

    # Total reward calculation with scaled contributions
    total_reward = 0.5 * distance_reward + 0.3 * door_opening_reward - 0.2 * movement_penalty

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty
    }
    return total_reward, reward_dict
