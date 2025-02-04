@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Enhanced Distance Reward with Recalibrated Temperature
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.5  # Re-calibrated for effectiveness
    distance_reward = torch.exp(-distance_to_handle / distance_reward_temperature)

    # Scaled Door Opening Reward
    door_opening_reward_temperature = 0.5  # Reduce influence
    door_opening_reward = torch.exp(cabinet_dof_pos[:, 3] / door_opening_reward_temperature) - 1.0

    # New Fine Movement Penalty for Smoothly Open Movement
    movement_smoothness_penalty_temperature = 0.5  # Temperature for reaction
    movement_smoothness_penalty = torch.exp(-torch.abs(cabinet_dof_vel[:, 3]) / movement_smoothness_penalty_temperature)

    # New Alignment Reward to Encourage Correct Drawer Alignment
    alignment_reward_temperature = 1.0  # Added for drawer alignment
    alignment_offset = torch.norm(drawer_grasp_pos - franka_grasp_pos, p=1, dim=-1)  # Example alignment metric
    alignment_reward = torch.exp(-alignment_offset / alignment_reward_temperature)

    # Total reward calculation with combined and balanced weightings
    total_reward = 0.3 * distance_reward + 0.4 * door_opening_reward - 0.3 * movement_smoothness_penalty + 0.2 * alignment_reward

    # Detailed reward breakdown dictionary
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'movement_smoothness_penalty': movement_smoothness_penalty,
        'alignment_reward': alignment_reward
    }
    
    return total_reward, reward_dict
