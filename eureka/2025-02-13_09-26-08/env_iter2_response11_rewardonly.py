@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # A greater emphasis on minimizing the distance to the handle using a lower temperature
    dist_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Lower to increase sensitivity to changes
    dist_reward = torch.exp(-dist_to_handle / temperature_distance)

    # Encourage movement towards the ideal location of a fully opened door 
    door_target_open_pos = torch.tensor(1.0, device=cabinet_dof_pos.device)  # Assuming full open value is 1.0
    opening_contribution = cabinet_dof_pos[:, 3] / door_target_open_pos
    opening_restored_reward = torch.clamp(opening_contribution, min=0.0, max=1.0) * 3.0  # Increase scaling

    # Improve velocity reward to further incentivize movement
    temperature_velocity = 0.02  # New temperature for sensitivity to speed change
    velocity_reward = torch.exp(cabinet_dof_vel[:, 3] / temperature_velocity) - 1.0  # Exponential function

    # Combine reward components, considering adjustments
    total_reward = 0.3 * dist_reward + 0.5 * opening_restored_reward + 0.2 * velocity_reward

    # Compile each component in a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
