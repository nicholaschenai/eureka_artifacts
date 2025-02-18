@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Restructured distance reward to encourage handle proximity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.05  # Lower temperature for greater reward differentiation
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Revised reward component for door opening focus
    door_angle = cabinet_dof_pos[:, 3].clamp(min=0.0)
    temperature_opening = 0.1
    opening_reward = torch.exp(door_angle / temperature_opening)

    # Capped velocity component for maintaining controlled motion
    door_velocity = cabinet_dof_vel[:, 3]
    velocity_limit = 1.0  # Reducing the influence scale
    velocity_reward = torch.exp(torch.clamp(door_velocity, max=velocity_limit) / temperature_opening)

    # Composite reward computation
    total_reward = 2.0 * dist_reward + 3.0 * opening_reward + 0.5 * velocity_reward

    # Components dictionary for insights
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
