@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Enhanced distance reward to strongly encourage proximity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Lowered temperature for stronger gradient
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Moderately scaled opening reward with adjusted sensitivity
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_opening = 0.2  # Adjusted temperature for moderate reward
    opening_restored_reward = torch.exp(torch.abs(door_open_value) / temperature_opening) * 0.5  # Reduced strength

    # Revised velocity reward to balance influence
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 2.0  # Increasing temperature for softening effect
    velocity_reward = torch.exp(-door_velocity / temperature_velocity)

    # Compose the total reward with new weights
    total_reward = 3.0 * dist_reward + 1.0 * opening_restored_reward + 1.0 * velocity_reward

    # Dictionary with individual components for analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
