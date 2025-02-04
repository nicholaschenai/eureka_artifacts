@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = franka_grasp_pos.device

    # Rewritten Distance Reward
    target_distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5  # Make it more sensitive to changes
    distance_reward = torch.exp(-target_distance / distance_reward_temperature)

    # Rescaled Door Opening Reward
    drawer_open_position = cabinet_dof_pos[:, 3]
    door_opening_reward_temperature = 5.0  # Increased to balance scale
    max_open_pos = 1.0  # Assuming a realistic maximum
    door_opening_reward = (drawer_open_position / max_open_pos).clamp(max=1.0)

    # Enhanced Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.5
    movement_penalty = -torch.exp(torch.abs(door_velocity) / movement_penalty_temperature) + 1.0  # Normalized negative impact

    # Total reward calculation
    total_reward = 0.4 * distance_reward + 0.4 * door_opening_reward + 0.2 * movement_penalty

    # Reward breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty
    }
    
    return total_reward, reward_dict
