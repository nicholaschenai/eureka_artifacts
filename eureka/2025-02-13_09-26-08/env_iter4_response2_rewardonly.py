@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Enhanced Distance Reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1  # Reduced to boost sensitivity
    dist_reward = 1.0 / (1.0 + distance_to_handle / 0.1)  # New transformation for sharper gradients

    # Normalized Door Opening Reward
    door_open_value = torch.abs(cabinet_dof_pos[:, 3])
    temperature_opening = 0.5  # Increased for less aggression
    opening_restored_reward = door_open_value / (1.0 + door_open_value / temperature_opening) 

    # Velocity Reward Rewritten
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    velocity_reward = 1.0 - torch.min(door_velocity / 5.0, torch.tensor(1.0, device=door_velocity.device))  # Normalize to 0-1

    # Total reward composition respecting newly added rewards
    total_reward = 1.5 * dist_reward + 1.0 * opening_restored_reward + 0.5 * velocity_reward

    # Include individual components for diagnostics and improvement tracking
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
