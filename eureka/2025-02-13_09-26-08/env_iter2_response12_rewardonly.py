@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor,
    cabinet_dof_vel: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = franka_grasp_pos.device

    # Reward component for reducing distance to the handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_dist = 0.3  # Adjusted for more meaningful reward
    dist_reward = torch.exp(-distance_to_handle / temperature_dist)

    # Reward for opening direction progress
    door_opening_progress = cabinet_dof_pos[:, 3].clamp(min=0.0)
    opening_progress_reward = torch.tanh(door_opening_progress * 5.0)  # Scaled for better gradient

    # Modify velocity reward: focus on positive velocity and scale it
    velocity_reward = (cabinet_dof_vel[:, 3].clamp(min=0.0)) * 0.5

    # Combine all components, weighted appropriately
    total_reward = 1.0 * dist_reward + 2.0 * opening_progress_reward + 1.0 * velocity_reward

    # Dictionary of reward components
    reward_components = {
        "dist_reward": dist_reward,
        "opening_progress_reward": opening_progress_reward,
        "velocity_reward": velocity_reward
    }

    return total_reward, reward_components
