@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Improved Distance Reward to encourage proximity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Increased sensitivity
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Scaled Door Opening Reward to reduce its dominant impact
    door_opening_reward_temperature = 1.0  # Increased temperature to scale down the magnitude
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature)

    # Enhanced Movement Penalty to encourage interaction
    door_velocity = cabinet_dof_vel[:, 3]
    movement_penalty_temperature = 0.1  # Reduced temperature for higher penalty sensitivity
    movement_penalty = torch.exp(-torch.abs(door_velocity) / movement_penalty_temperature)
    
    # New penalty: Time penalty to promote quicker task completion
    time_penalty_temperature = 0.2
    time_penalty = torch.exp(cabinet_dof_pos[:, 3] / time_penalty_temperature)  # Assumes progression reflects task progression
    
    # Total reward calculation with reassessed weighting
    total_reward = 0.4 * distance_reward + 0.3 * door_opening_reward - 0.2 * movement_penalty - 0.1 * time_penalty

    # Return the final reward and component breakdown
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_penalty': movement_penalty,
        'time_penalty': time_penalty
    }
    return total_reward, reward_dict
