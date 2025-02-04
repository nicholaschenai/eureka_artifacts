@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Increased sensitivity
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Rescaled Door Opening Reward
    door_opening_threshold = 0.5  # Assuming a normal threshold for a fully opened door
    door_opening_condition = cabinet_dof_pos[:, 3] > door_opening_threshold
    door_opening_reward_temperature = 0.5
    door_opening_reward = torch.where(door_opening_condition,
                                      torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature),
                                      torch.tensor(0.0, device=franka_grasp_pos.device))

    # Adjusted Movement Penalty
    movement_penalty_temperature = 0.5
    movement_penalty = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_penalty_temperature)

    # New Penalty for Time Steps (Discourage long episodes)
    time_penalty_scale = 0.01
    time_penalty = time_penalty_scale * torch.ones_like(distance_to_handle)  # Constant step penalty
    
    # Task completion bonus (additional incentive when the door is fully opened)
    completion_bonus = torch.where(door_opening_condition, torch.tensor(1.0, device=franka_grasp_pos.device), torch.tensor(0.0, device=franka_grasp_pos.device))

    # Total reward calculation
    total_reward = (0.3 * distance_reward + 
                    0.5 * door_opening_reward - 
                    0.2 * movement_penalty - 
                    0.2 * time_penalty + 
                    completion_bonus)

    # Return the total reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'time_penalty': time_penalty,
        'completion_bonus': completion_bonus
    }
    return total_reward, reward_dict
