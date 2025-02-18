@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Enhanced distance reward with scaling
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Lowered temperature for sharper scaling
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)
    
    # Modify the opening restored reward to prevent overshadowing
    opening_restored = cabinet_dof_pos[:, 3].clamp(min=0.0)
    opening_restored_reward = opening_restored

    # Rework velocity reward to increase the influence
    velocity_threshold = 0.05  # Threshold for minimum meaningful velocity
    velocity_reward = torch.where(
        cabinet_dof_vel[:, 3] > velocity_threshold,
        cabinet_dof_vel[:, 3],
        torch.tensor(0.0, device=cabinet_dof_vel.device)
    )
    
    # Additional reward for reaching the drawer handle
    reach_threshold = 0.02
    reach_reward = torch.where(
        distance_to_handle < reach_threshold,
        torch.tensor(1.0, device=distance_to_handle.device),
        torch.tensor(0.0, device=distance_to_handle.device)
    )

    # Combine small weights to balance all components
    total_reward = 0.3 * dist_reward + 0.5 * opening_restored_reward + 0.2 * velocity_reward + 0.3 * reach_reward

    # Reward component dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward,
        "reach_reward": reach_reward
    }

    return total_reward, reward_components
