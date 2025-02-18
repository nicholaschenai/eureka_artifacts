@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Re-designed proximity reward to emphasize grasp proximity
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1) + 1e-6  # Avoid division by zero
    temperature_distance = 0.1  # Increased sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance) * 5.0

    # Re-scaled opening reward 
    door_open_value = cabinet_dof_pos[:, 3]
    opening_restored_reward = torch.clamp(door_open_value, min=0.0, max=0.5) * 10.0  # Scaled for importance but more reasonable

    # Total reward emphasizing task success: approach and open
    total_reward = 1.0 * dist_reward + 1.0 * opening_restored_reward

    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward
    }

    return total_reward, reward_components
