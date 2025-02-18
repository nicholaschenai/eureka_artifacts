@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Improved reward for reducing distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Reduced temperature to increase sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Enhanced reward for achieving and maintaining door opening
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)  # Focus on the positive (opening)
    opening_restored_reward = torch.tanh(door_open_value) * 3.0  # Emphasize through tanh transformation

    # Amplifying the velocity reward for consistent door movement direction
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.1
    velocity_reward = torch.exp(door_velocity / temperature_velocity) * 0.5  # Adjusted scaling for balance

    # Compose the total reward
    total_reward = 1.0 * dist_reward + 2.0 * opening_restored_reward + 0.5 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
