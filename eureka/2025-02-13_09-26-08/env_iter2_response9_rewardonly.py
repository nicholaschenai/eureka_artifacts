@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor,
    drawer_grasp_pos: torch.Tensor,
    franka_grasp_rot: torch.Tensor,
    drawer_grasp_rot: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
  
    # Reward component for minimizing the distance between the robot hand and the drawer handle
    distance_to_handle = torch.norm(franka_grasp_pos - drawer_grasp_pos, dim=-1)
    temperature_distance = 0.1  # Lowered to increase sensitivity
    dist_reward = torch.exp(-distance_to_handle / temperature_distance)
    
    # Reward for any motion that indicates the door is open beyond a small threshold
    opening_restored_threshold = 0.02  # Open door threshold
    door_opened = torch.gt(cabinet_dof_pos[:, 3], opening_restored_threshold).float()
    opening_restored_reward = door_opened * 0.5

    # Simplifying the reward mechanism by re-focusing on hand alignment rather than velocity
    alignment_error = torch.norm(torch.cross(franka_grasp_rot, drawer_grasp_rot), dim=-1)
    temperature_alignment = 0.1
    alignment_reward = torch.exp(-alignment_error / temperature_alignment)

    # Combine all reward components with adjusted scaling
    total_reward = (1.0 * dist_reward + 2.0 * opening_restored_reward + 1.5 * alignment_reward)

    # Collect each individual component in a dictionary
    reward_components = {
        "dist_reward": dist_reward,
        "opening_restored_reward": opening_restored_reward,
        "alignment_reward": alignment_reward
    }
    
    return total_reward, reward_components
