@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Re-scaled and re-designed distance reward to encourage grasping
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6  # Ensure no division by zero
    temperature_distance = 0.2  # Adjusted for more sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 2.0  # Scaled up

    # Stronger incentive for door opening
    door_open_value = cabinet_dof_pos[:, 3]  # Range modification - focus on opening
    temperature_opening = 0.1
    opening_restored_reward = torch.exp(torch.abs(door_open_value) / temperature_opening) * 5.0  # Greater differentiation and scaling

    # Rebalanced velocity reward
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 1.0  # Greater temperature for finer gradients
    velocity_reward = torch.exp(-door_velocity / temperature_velocity)  # Brought down for balance

    # Compose the total reward
    total_reward = 1.0 * dist_reward + 2.0 * opening_restored_reward + 1.0 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
