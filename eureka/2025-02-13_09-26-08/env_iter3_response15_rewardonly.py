@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Component for minimizing distance to handle
    temperature_distance = 0.05  # Lower temperature for finer gradient
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Component for rewarding door opening, and rescaling for greater impact
    door_open_value = cabinet_dof_pos[:, 3]
    temperature_opening = 0.5
    opening_restored_reward = torch.exp(door_open_value / temperature_opening)

    # Encouraging positive velocity specifically towards opening
    door_velocity = cabinet_dof_vel[:, 3]
    positive_velocity = torch.clamp(door_velocity, min=0.0)  # Only reward positive (opening) velocity
    temperature_velocity = 0.05
    velocity_reward = torch.exp(positive_velocity / temperature_velocity)

    # Normalizing reward components to be of similar scale
    dist_reward_scaled = 0.5 * dist_reward
    opening_reward_scaled = 2.0 * opening_restored_reward
    velocity_reward_scaled = 0.5 * velocity_reward

    # Balanced total reward composition
    total_reward = dist_reward_scaled + opening_reward_scaled + velocity_reward_scaled

    # Collecting individual reward components for analysis
    reward_components = {
        "dist_reward": dist_reward_scaled,
        "opening_restored_reward": opening_reward_scaled,
        "velocity_reward": velocity_reward_scaled
    }

    return total_reward, reward_components
