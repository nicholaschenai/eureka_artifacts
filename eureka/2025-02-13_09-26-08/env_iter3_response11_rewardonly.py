@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjusted reward for reducing distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.2  # Increased temperature for broader range and sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)
    
    # Reformulated reward for opening the door
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)  # Focus on the positive (opening)
    temperature_opening = 0.5  # Suitable temperature for normalization
    opening_restored_reward = torch.exp(door_open_value / temperature_opening)  # Exponential to emphasize opening
    
    # Decreased scale for the velocity reward for balance
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.5  # Adjusted for appropriate impact
    velocity_reward = torch.tanh(door_velocity / temperature_velocity)  # Using tanh to maintain range

    # Aggregate total reward
    total_reward = 1.5 * dist_reward + 2.5 * opening_restored_reward + 0.3 * velocity_reward

    # Return individual components in a dictionary for analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
