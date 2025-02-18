@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Improving reward for proximity to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.2  # Adjusted temperature to be more sensitive
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 5.0  # Adjusted scaling

    # Overhaul reward for opening the drawer
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)
    temperature_open = 1.0
    opening_reward = torch.exp(door_open_value / temperature_open) * 3.0  # Applying exp for better distinction

    # Normalizing the velocity reward for balance
    door_velocity = cabinet_dof_vel[:, 3].abs()
    temperature_velocity = 1.0
    velocity_reward = torch.tanh(door_velocity / temperature_velocity) * 1.0  # Tan-h for stable scaling

    # Compose the total reward
    total_reward = 2.0 * dist_reward + 3.0 * opening_reward + 1.0 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
