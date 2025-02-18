@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # New design for distance reward with improved sensitivity and range
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Further reduced to increase sensitivity
    dist_reward = torch.clamp(1.0 - (distance_to_handle / (torch.tensor(0.5).to(franka_grasp_pos.device))), 0, 1)
    
    # Revised door opening reward that more effectively incentivizes opening beyond an initial range
    door_open_value = cabinet_dof_pos[:, 3]
    opening_threshold = 0.2
    opening_restored_reward = torch.where(door_open_value > opening_threshold, 
                                          torch.tanh((door_open_value - opening_threshold) * 5.0), 
                                          torch.tensor(0.0, device=franka_grasp_pos.device))

    # Properly rescaled velocity to prevent overshadowing
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.2
    velocity_reward = torch.clamp(door_velocity / (temperature_velocity + 1e-6), 0, 1.0) * 0.1  # Rescaled to 0-1 range
    
    # Total reward calculation balanced across components
    total_reward = 2.0 * dist_reward + 3.0 * opening_restored_reward + 1.0 * velocity_reward

    # Dictionary of individual components for analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
