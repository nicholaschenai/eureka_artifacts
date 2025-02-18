@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Stronger encouragement to reduce the hand-target distance
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Make distance reward more sensitive
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 2.0

    # More aggressive incentive for the door being opened further
    door_open_value = cabinet_dof_pos[:, 3]
    temp_opening = 0.2  # Adjust temperature for better sensitivity
    opening_reward = torch.sigmoid(door_open_value / temp_opening) * 5.0

    # Penalize high velocity unless positive towards opening
    door_velocity = cabinet_dof_vel[:, 3]
    velocity_penalty = torch.relu(-door_velocity) / 5.0

    # Compose the total reward
    total_reward = 1.5 * dist_reward + 2.5 * opening_reward - 0.5 * velocity_penalty

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_penalty": velocity_penalty
    }
    
    return total_reward, reward_components
