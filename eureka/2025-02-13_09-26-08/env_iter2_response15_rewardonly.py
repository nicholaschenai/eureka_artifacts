@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Adjusted Distance Reward transformation for emphasis on proximity
    distance = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # More aggressive shaping to emphasize closing in
    dist_reward = torch.exp(-distance / temperature_distance)
    
    # Altered Door Opening Reward to ensure only substantial openings are rewarded
    opening_amount = cabinet_dof_pos[:, 3]
    open_threshold = 0.2  # Reward only if opening is beyond the threshold
    opening_restored_reward = (opening_amount > open_threshold).float() * 0.5

    # New Velocity Component: Emphasis on active movement
    velocity_threshold = 0.05
    velocity_reward = torch.where(cabinet_dof_vel[:, 3] > velocity_threshold, cabinet_dof_vel[:, 3] * 2.0, torch.tensor(0.0, device=cabinet_dof_vel.device))

    # Combine all reward components with revised scaling
    total_reward = 1.0 * dist_reward + 1.0 * opening_restored_reward + 0.5 * velocity_reward

    # Collect all components into a dictionary for analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
