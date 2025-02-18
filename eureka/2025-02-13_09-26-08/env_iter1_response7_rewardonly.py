@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.2  # Adjusted temperature parameter for scaling
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Reward for any motion that indicates the door is opening
    opening_restored = cabinet_dof_pos[:, 3].clamp(min=0.0)
    opening_restored_reward = opening_restored * 2.0  # Multiply to scale the effect

    # Velocity indicating door movement in the opening direction
    velocity_reward = torch.clamp(cabinet_dof_vel[:, 3], min=0.0) * 1.5  # Adjusted reward scaling

    # Combine all reward components
    total_reward = 0.5 * dist_reward + 1.0 * opening_restored_reward + 1.0 * velocity_reward

    # Collect each individual component in a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
