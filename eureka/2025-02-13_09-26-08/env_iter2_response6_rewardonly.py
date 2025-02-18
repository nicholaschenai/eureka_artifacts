@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Distance reward with enhanced sensitivity using a temperature parameter
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.5  # A lower temperature to intensify reward sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Enhance the diversity of states by scaling door position for significant reward
    opening_restored = cabinet_dof_pos[:, 3].clamp(min=0.0)
    opening_reward_scale = 5.0  # A higher scale to encourage continuous door opening
    opening_restored_reward = opening_restored * opening_reward_scale

    # Re-evaluated velocity reward component to reflect dynamic interactions
    dynamic_velocity_threshold = 0.05  # Encourage movement above a minor threshold
    moving_velocity_reward = torch.where(
        cabinet_dof_vel[:, 3] > dynamic_velocity_threshold,
        cabinet_dof_vel[:, 3] * 2.0,
        torch.tensor(0.0, device=cabinet_dof_vel.device)
    )

    # Total reward places stronger focus on door opening actions
    total_reward = 0.8 * dist_reward + 1.2 * opening_restored_reward + 0.2 * moving_velocity_reward

    # Dictionary to track all reward components
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "moving_velocity_reward": moving_velocity_reward
    }

    return total_reward, reward_components
