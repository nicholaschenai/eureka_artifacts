@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # A more aggressive distance component reward
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Aggressively scaled temperature parameter
    dist_reward = 1.0 - torch.tanh(distance_to_handle / temperature_distance)

    # Continued focus on the extent of door opening
    opening_reward = torch.clamp(cabinet_dof_pos[:, 3], min=0.0, max=1.0)

    # Amplified reward for opening velocity; it encourages consistent movement
    velocity_magnitude = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.01
    velocity_reward = torch.sign(cabinet_dof_vel[:, 3]) * torch.exp(velocity_magnitude / temperature_velocity)

    # Combine all reward components with re-balanced weighting
    total_reward = 0.3 * dist_reward + 0.5 * opening_reward + 0.2 * velocity_reward

    # Collect each individual component in a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
