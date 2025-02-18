@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # New - Improved distance reward for precise encouragement of grasping
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1  # More sensitive temperature
    dist_reward = 1.0 / (distance_to_handle + temperature_distance * 0.5)  # Inverse approach for diminishing reward

    # Re-scaled opening reward to reduce skew
    door_open_value = cabinet_dof_pos[:, 3]
    normalized_open_value = torch.clamp(door_open_value / 1.0, -1.0, 1.0)  # Ensure range control
    temperature_opening = 0.1
    opening_restored_reward = normalized_open_value * 3.0  # More moderate scaling

    # Redesigned velocity reward for better gradation
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.5  # Sharper differentiation
    velocity_reward = torch.exp(-door_velocity / temperature_velocity) * 0.5  # Reduced weight

    # Total reward computation ensuring balanced contribution
    total_reward = 0.5 * dist_reward + 1.5 * opening_restored_reward + 0.5 * velocity_reward

    # Dictionary to enable deeper analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
