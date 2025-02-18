@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Revised reward for minimizing distance to handle
    dist_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.5  # Adjusted for better scaling
    dist_reward = torch.exp(-dist_to_handle / temperature_distance)

    # Enhanced reward for opening the door
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0)
    temperature_opening = 0.5  # New temperature for scaling
    opening_reward = torch.tanh(door_open_value / temperature_opening) * 2.0  # Adjust scaling for emphasis

    # Rescaled and revised velocity reward
    door_velocity = cabinet_dof_vel[:, 3]
    velocity_direction = torch.sign(door_velocity)  # Encouraging only positive opening
    temperature_velocity = 1.0  # New temperature for better control
    velocity_reward = torch.where(velocity_direction > 0, (torch.exp(door_velocity / temperature_velocity) - 1.0), torch.tensor(0.0, device=franka_grasp_pos.device)) * 0.1  # Downscaled to balance

    # Compose total reward
    total_reward = 1.5 * dist_reward + 3.0 * opening_reward + 0.5 * velocity_reward

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
