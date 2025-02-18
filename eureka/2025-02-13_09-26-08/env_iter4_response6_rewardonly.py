@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Revised Distance Reward: Enhance sensitivity and scale for better control
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    dist_temp = 0.4  # More aggressive
    dist_reward = 1.0 - torch.tanh(distance_to_handle / dist_temp)

    # Rescaled Opening Reward: Normalize so it doesn't overshadow others 
    door_open_value = cabinet_dof_pos[:, 3]
    opening_temp = 0.3  # Lower temperature for better precision scaling
    opening_reward = torch.sigmoid(door_open_value / opening_temp) * 2.0

    # Optimization achieved by prioritizing new combined task-focused learning
    total_reward = 1.5 * dist_reward + 2.5 * opening_reward

    reward_components = {
        "dist_reward": dist_reward,
        "opening_reward": opening_reward
    }

    return total_reward, reward_components
