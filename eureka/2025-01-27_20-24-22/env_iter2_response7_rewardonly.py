@torch.jit.script
def compute_reward(franka_grasp_pos: torch.Tensor, 
                   drawer_grasp_pos: torch.Tensor, 
                   cabinet_dof_pos: torch.Tensor, 
                   cabinet_dof_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Revised Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
    distance_reward_temperature = 0.2  # Make it more sensitive
    distance_reward = 1.0 / (1.0 + distance_to_handle / distance_reward_temperature)

    # Re-scaled Door Opening Reward
    door_opening_reward_temperature = 0.5  # Adjust to balance influence
    door_position_opening = torch.clamp(cabinet_dof_pos[:, 3], 0.0, 1.0)
    door_opening_reward = torch.exp(door_position_opening / door_opening_reward_temperature)

    # Simplified Penalty for inactivity
    inactivity_penalty_temperature = 0.1
    inactivity_penalty = 1.0 / (1.0 + torch.abs(cabinet_dof_vel[:, 3]) / inactivity_penalty_temperature)

    # Total reward re-calibrated
    total_reward = 0.4 * distance_reward + 0.4 * door_opening_reward - 0.2 * inactivity_penalty

    # Return rewards with updated contributions
    reward_dict = {
        'distance_reward': distance_reward,
        'door_opening_reward': door_opening_reward,
        'inactivity_penalty': inactivity_penalty
    }
    
    return total_reward, reward_dict
