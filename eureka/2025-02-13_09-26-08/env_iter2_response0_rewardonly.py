@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Improved distance reward with a higher temperature and exponentiation
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.5
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)
    
    # Door opening reward emphasizing the amount of displacement
    door_opening = cabinet_dof_pos[:, 3]
    open_scaling = 3.0  # More emphasis on opening
    open_reward = door_opening * open_scaling
    
    # Velocity reward re-calibrated with higher impact, focusing on genuine motion
    velocity_scalar = 2.0
    velocity_reward = torch.relu(cabinet_dof_vel[:, 3]) * velocity_scalar

    # Total reward as a weighted sum of all components, adjusting weights for improved balance
    total_reward = (
        0.75 * dist_reward +  # Emphasizing initial approach to handle
        1.0 * open_reward +   # Strong emphasis on opening motion
        0.5 * velocity_reward  # Moderately scaled encouraging consistent motion
    )

    # Compile individual reward components into a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "open_reward": open_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
