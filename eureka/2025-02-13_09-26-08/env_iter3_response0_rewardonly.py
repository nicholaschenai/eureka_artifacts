@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Stronger transformation for the distance reward to encourage proximity to handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Decreased temperature to make reward more sensitive
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 2.0

    # Revised opening reward for door position, ensuring a strong incentive for progress
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)
    temperature_opening = 1.0  # Higher sensitivity to opening progress
    opening_restored_reward = torch.exp(door_open_value / temperature_opening) * 5.0

    # Re-scaled velocity reward to contribute proportionately
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.2  # Adjusted temperature for smoother scaling
    velocity_reward = (torch.exp(door_velocity / temperature_velocity) - 1.0) * 0.1  # Reduced influence

    # Compose the total reward, emphasizing balance across components
    total_reward = dist_reward + opening_restored_reward + velocity_reward

    # Include individual components in results for diagnostics
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
