@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Decreased temperature for sharper gradient
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Reward for the extent to which the door opening has been restored (more weight)
    opening_restored = cabinet_dof_pos[:, 3].clamp(min=0.0)
    opening_restored_reward = opening_restored * 3.0  # Increased scaling factor for more emphasis

    # Velocity indicating door movement in the opening direction
    velocity_magnitude = torch.abs(cabinet_dof_vel[:, 3])  # Consider both directions
    velocity_reward = velocity_magnitude * 5.0  # Increase to reward active velocity

    # Combine all reward components with recalibrated factors
    total_reward = 1.0 * dist_reward + 1.5 * opening_restored_reward + 0.5 * velocity_reward

    # Collect each individual component in a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
