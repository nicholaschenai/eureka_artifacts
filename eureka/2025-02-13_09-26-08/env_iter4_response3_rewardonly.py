@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjusted distance reward with widened effective range
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.5
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Rescaled opening reward for balance
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_opening = 1.0
    opening_restored_reward = torch.sigmoid(door_open_value / temperature_opening) * 3.0

    # Recalibrated velocity reward introducing directional movement bonus
    door_velocity = cabinet_dof_vel[:, 3]
    directional_bonus = torch.where(door_velocity > 0, torch.tensor(1.0, device=door_velocity.device), torch.tensor(0.0, device=door_velocity.device))
    temperature_velocity = 0.5
    velocity_reward = torch.exp(-door_velocity.abs() / temperature_velocity) + directional_bonus

    # Composite total reward
    total_reward = 1.5 * dist_reward + 3.0 * opening_restored_reward + 1.0 * velocity_reward

    # Component dictionary for debugging and analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
