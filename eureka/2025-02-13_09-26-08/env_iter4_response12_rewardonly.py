@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Enhanced and re-scaled distance reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1  # Increased precision
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 3.0  # Scaled up for prominence

    # Down-scaled door opening reward
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_opening = 0.5
    opening_restored_reward = torch.exp(torch.abs(door_open_value) / temperature_opening) * 0.5  # Reduced impact

    # Revised and sensitive velocity reward
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.5
    velocity_reward = -door_velocity * 0.1  # Penalized for high velocities

    # Early completion reward
    max_episode_length = 500.0
    completion_reward = ((max_episode_length - torch.norm(cabinet_dof_pos[:, 3])) / max_episode_length) * 2.0

    # Compose total reward
    total_reward = 1.5 * dist_reward + 0.8 * opening_restored_reward + 1.0 * velocity_reward + 1.0 * completion_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward,
        "completion_reward": completion_reward
    }

    return total_reward, reward_components
