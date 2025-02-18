@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjusted distance reward to emphasize proximity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1  # Adjusted for greater sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)  # No additional scaling to maintain balance

    # Scaled down opening reward to ensure balanced contribution
    door_open_value = torch.clamp(cabinet_dof_pos[:, 3], 0.0, 1.0)  # Ensure within [0, 1] range
    temperature_opening = 0.2
    opening_restored_reward = torch.exp(door_open_value / temperature_opening)  # Remove unnecessary multiplication

    # Revised velocity reward for more meaningful impact and discourage excessive velocities
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.5  # Increased temperature for better highlighting velocity impact
    velocity_reward = torch.exp(-door_velocity / temperature_velocity)

    # New encouraging reward to motivate complete actions
    task_completion_reward = torch.where(door_open_value > 0.9, torch.tensor(1.0, device=door_open_value.device), torch.tensor(0.0, device=door_open_value.device))

    # Compose the total reward
    total_reward = 0.5 * dist_reward + 1.0 * opening_restored_reward + 0.5 * velocity_reward + 5.0 * task_completion_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward,
        "task_completion_reward": task_completion_reward
    }

    return total_reward, reward_components
