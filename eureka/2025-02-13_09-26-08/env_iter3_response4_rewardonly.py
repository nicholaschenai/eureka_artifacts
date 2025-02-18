@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Reward for reducing the distance to the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.2  # Adjusting temperature for better reward shaping
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 3.0

    # Improved reward component for door opening
    door_open_value = cabinet_dof_pos[:, 3].clamp(min=0.0, max=1.0)  # Added clamping for stability
    opening_reward = torch.tanh(door_open_value) * 5.0  # Enhanced scaling to emphasize door opening

    # Adjusting velocity reward to have balanced magnitude
    door_velocity = cabinet_dof_vel[:, 3]
    temperature_velocity = 0.5
    velocity_reward = (1.0 / (1.0 + torch.exp(-door_velocity / temperature_velocity))) * 2.0

    # Combine the total reward
    total_reward = 0.5 * dist_reward + 0.3 * opening_reward + 0.2 * velocity_reward

    # Dictionary with individual components for monitoring
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
