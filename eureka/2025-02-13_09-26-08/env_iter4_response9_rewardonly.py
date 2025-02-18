@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Enhanced distance reward for encouraging precision in grasping
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Reduced for more sensitivity in grasping
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 3.0  # Slightly larger scale

    # Rescaled opening reward
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_opening = 0.5  # Higher temperature to reduce overwhelming impact
    opening_restored_reward = torch.sigmoid(door_open_value / temperature_opening) * 1.5  # Diminished rate of increase

    # Remove or drastically reduce velocity reward to focus on essential task components
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.5  # Adjusted temperature for residual effects
    velocity_reward = torch.exp(-door_velocity / temperature_velocity) * 0.5  # Much diminished in effect
    
    # Compose total reward
    total_reward = 1.5 * dist_reward + 1.0 * opening_restored_reward + 0.5 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
