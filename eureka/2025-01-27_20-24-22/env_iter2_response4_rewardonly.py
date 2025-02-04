@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5
    distance_reward = 1.0 / (1.0 + torch.exp(distance_to_handle / distance_reward_temperature))
    
    # Rescaled Door Opening Reward
    door_opening_reward_temperature = 0.5
    door_opening_reward = torch.tanh(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Enhanced Movement Penalty
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.5
    movement_penalty = -torch.tanh(torch.abs(door_velocity) / movement_penalty_temperature)

    # Introduced Completion Bonus for Efficiency
    completion_bonus_temperature = 0.1
    completion_bonus = torch.where(cabinet_dof_pos[:, 3] > 0.9, torch.tensor(1.0), torch.tensor(0.0))
    completion_bonus_adjusted = torch.exp(completion_bonus / completion_bonus_temperature)

    # Combined Total Reward with adjusted scales
    total_reward = (0.4 * distance_reward) + (0.3 * door_opening_reward) + (0.2 * completion_bonus_adjusted) + (0.1 * movement_penalty)

    # Reward breakdown for analysis
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'completion_bonus': completion_bonus_adjusted
    }

    return total_reward, reward_dict
