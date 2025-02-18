@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjusted distance reward to be more impactful
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1  # Lowered for a sharper differentiation
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 3.0  # Raised scaling

    # Re-scaled opening reward to mitigate its dominance
    door_open_value = cabinet_dof_pos[:, 3]
    temp_opening = 0.2
    opening_restored_reward = torch.exp(torch.abs(door_open_value) / temp_opening) * 1.0  # Scaled down

    # Completely re-worked velocity reward for more impact
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.5
    velocity_direction_reward = torch.where(door_velocity > 0, torch.tensor(1.0, device=door_velocity.device), torch.tensor(-1.0, device=door_velocity.device))
    velocity_reward = velocity_direction_reward * torch.exp(-torch.abs(door_velocity) / temperature_velocity)

    # Sum all rewards with adjusted weights
    total_reward = 2.0 * dist_reward + 1.0 * opening_restored_reward + 2.0 * velocity_reward

    # Detailed breakdown
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
