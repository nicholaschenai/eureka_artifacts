@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Rewriting distance reward for better encouragement
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Lower temperature for sharper rewards
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)
    
    # Enhanced opening restored reward combined with distance
    opening_target = torch.ones_like(cabinet_dof_pos[:, 3])  # Assuming 1.0 as fully open position
    opening_error = opening_target - cabinet_dof_pos[:, 3]
    temperature_opening = 0.1
    opening_reward = torch.exp(-torch.abs(opening_error) / temperature_opening) * 2.0  # Stronger weight on the door state
    
    # Rescaled velocity reward for balance
    velocity_penalty = cabinet_dof_vel[:, 3].clamp(max=0)  # Only encourage non-negative velocity
    temperature_velocity = 0.2
    velocity_reward = torch.exp(velocity_penalty / temperature_velocity)
    
    # Compose the total reward
    total_reward = 1.0 * dist_reward + 2.0 * opening_reward - 0.5 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
