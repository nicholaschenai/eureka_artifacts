@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor,
    max_episode_length: torch.Tensor,
    curr_episode_step: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Enhanced distance reward for effective grasping
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6
    temperature_distance = 0.05  # Decreased for higher sensitivity and engagement
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 4.0  # Increased influence

    # Re-scaled opening reward to balance significance and include a success incentive
    door_open_value = cabinet_dof_pos[:, 3]
    opening_threshold = 0.5  # Define an arbitrary threshold for a successful open
    opening_success_bonus = torch.gt(door_open_value, opening_threshold).float() * 5.0  # Fixed bonus for surpassing the threshold
    temperature_opening = 1.0
    opening_restored_reward = torch.exp(door_open_value / temperature_opening) * 0.5 + opening_success_bonus  # Re-scaled

    # Re-evaluated velocity reward for better integration
    door_velocity = torch.abs(cabinet_dof_vel[:, 3])
    temperature_velocity = 0.5  # Adjusted for better reward shaping
    velocity_reward = torch.exp(-door_velocity / temperature_velocity) * 2.0  # Adjusted scale

    # New component for episode length penalty
    episode_penalty = torch.clamp(1.0 - (curr_episode_step / max_episode_length), min=0.0) * 2.0  # Penalize longer episodes

    # Compose the total reward
    total_reward = dist_reward + opening_restored_reward + velocity_reward + episode_penalty

    # Dictionary with individual components for detailed analysis
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward,
        "episode_penalty": episode_penalty
    }

    return total_reward, reward_components
