@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Improved and larger distance reward to promote proximity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1  # Increased sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 5.0  # Scaled up significantly

    # Adjusted door opening reward to be less overwhelming
    door_open_value = cabinet_dof_pos[:, 3]
    is_opening_penalty_threshold = 0.2  # Apply penalty if not opened wider than threshold
    opening_penalty = torch.where(door_open_value < is_opening_penalty_threshold, -1.0, 0.0)
    temperature_opening = 0.5
    opening_restored_reward = torch.exp(torch.abs(door_open_value) / temperature_opening) * 2.0 + opening_penalty

    # New velocity damping reward to manage stability
    stability_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_stability = 0.5
    stability_reward = torch.exp(-stability_velocity / temperature_stability) * 2.0  # Replaced with a focused task stability metric

    # Task completion bonus for achieving specific door position
    task_completion_bonus = torch.where(door_open_value > 0.5, 10.0, 0.0)  

    # Total reward calculation
    total_reward = dist_reward + opening_restored_reward + stability_reward + task_completion_bonus

    # Reward components dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "stability_reward": stability_reward,
        "task_completion_bonus": task_completion_bonus
    }

    return total_reward, reward_components
