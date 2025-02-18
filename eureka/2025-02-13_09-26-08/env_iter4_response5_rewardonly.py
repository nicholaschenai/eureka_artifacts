@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor,
    episode_length: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Re-designed distance reward to incentivize closer grasp
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.1  # Increased sensitivity
    dist_reward = 1.0 / (1.0 + distance_to_handle)  # More intuitive reward scale for proximity

    # Rescaled opening reward for proper balance
    door_open_value = torch.clamp(cabinet_dof_pos[:, 3], min=0.0, max=1.0)  # Assuming a normalized scale
    temperature_opening = 0.5
    opening_reward = torch.exp(door_open_value / temperature_opening) - 1.0  # Adjusted to reduce magnitude

    # Re-implemented velocity reward for variability
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.1
    velocity_reward = torch.exp(-door_velocity / temperature_velocity) - 1.0  # Making it more significant

    # Episode time cost to encourage faster task completion
    episode_penalty = torch.exp(-episode_length / 100.0)  # Penalty for longer episodes

    # Define the new total reward
    total_reward = 1.5 * dist_reward + 3.0 * opening_reward + velocity_reward + 0.5 * episode_penalty

    # Dictionary with individual components
    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward,
        "velocity_reward": velocity_reward,
        "episode_penalty": episode_penalty
    }

    return total_reward, reward_components
