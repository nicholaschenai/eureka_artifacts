@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor, 
    cabinet_dof_pos: torch.Tensor, 
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Revised reward component for minimizing the distance between the robot's hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Lowered temperature for sharper feedback
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 2.0  # Increased scaling

    # Reward for the degree the door is opened, emphasizing task completion
    goal_open_position = 0.8  # Assume goal is to open to 80%
    door_opening_progress = (cabinet_dof_pos[:, 3] / goal_open_position).clamp(min=0.0, max=1.0)
    opening_restored_reward = door_opening_progress * 3.0  # Prioritize opened percentage over other metrics

    # Consider de-emphasizing or removing velocity-based incentives due to limited impact
    velocity_reward = cabinet_dof_vel[:, 3].clamp(min=0.0) * 0.5  # Halved influence if retained

    # Combining all components into the total reward
    total_reward = 1.0 * dist_reward + 1.5 * opening_restored_reward + 0.5 * velocity_reward

    # Collect individual components in a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
