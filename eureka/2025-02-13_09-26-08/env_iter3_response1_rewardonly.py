@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # New distance reward: More sensitivity for precision
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Lower temperature to increase sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 2.0  # Emphasize more

    # Reformed door opening reward: Increasing encouragement with effective transformation
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)
    temperature_opening = 1.0
    opening_restored_reward = torch.exp(door_open_value / temperature_opening) - 1.0  # Normalize and boost effect

    # Re-scaled velocity reward: Normalize to balance with other components
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.5  # Tweak temperature for balance
    velocity_reward = torch.tanh(door_velocity / temperature_velocity)  # Use tanh to keep within -1 to 1

    # Construct the total reward
    total_reward = 1.5 * dist_reward + 2.5 * opening_restored_reward + 0.5 * velocity_reward

    # Dictionary with individual component rewards
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
