@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance Reward: Encouraging the agent to close the distance to the handle.
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.5
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)

    # Opening Reward: Rebalanced to integrate more naturally with other components.
    door_open_value = torch.abs(cabinet_dof_pos[:, 3])
    temperature_opening = 0.5
    opening_reward = torch.exp(door_open_value / temperature_opening)

    # New Component: Reward for maintaining drawer velocity within a moderate range to promote sustained opening.
    desired_velocity_low = 0.1
    desired_velocity_high = 0.5
    velocity_margin = torch.abs(cabinet_dof_vel[:, 3] - desired_velocity_low) + 1e-6
    velocity_margin = torch.clamp(velocity_margin, min=0, max=desired_velocity_high - desired_velocity_low)
    temperature_velocity = 0.5
    velocity_reward = torch.exp(-velocity_margin / temperature_velocity)

    # Total Reward: Harmoniously combining components for balanced learning.
    total_reward = 1.0 * dist_reward + 3.0 * opening_reward + 1.0 * velocity_reward

    # Store each component in the dict for detailed result analysis.
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
