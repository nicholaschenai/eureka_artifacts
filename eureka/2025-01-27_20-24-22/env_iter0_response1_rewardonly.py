@torch.jit.script
def compute_reward(
    franka_grasp_pos: torch.Tensor, 
    drawer_grasp_pos: torch.Tensor,
    franka_grasp_rot: torch.Tensor,
    drawer_grasp_rot: torch.Tensor,
    cabinet_dof_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute distances
    to_drawer_distance = torch.norm(drawer_grasp_pos - franka_grasp_pos, dim=-1)
    
    # Compute angular alignment (dot product between quaternions)
    alignment = torch.sum(franka_grasp_rot * drawer_grasp_rot, dim=-1)
    
    # Compute opening progress
    opening_progress = cabinet_dof_pos[:, 3]  # Assuming index 3 corresponds to the door joint

    # Normalize distances
    max_distance = 1.0  # Example, adjust as needed based on environment scale
    normalized_distance = to_drawer_distance / max_distance
    
    # Reward components
    dist_reward = torch.exp(-normalized_distance)
    alignment_reward = alignment  # Already between -1 and 1 due to quaternion properties
    progress_reward = opening_progress

    # Weights and temperature parameters
    dist_weight = 1.0
    align_weight = 0.1
    progress_weight = 1.0
    
    dist_temperature = 10.0  # Adjust the sensitivity of the distance reward

    # Total reward calculation
    total_reward = (dist_weight * torch.exp(-dist_temperature * normalized_distance) 
                    + align_weight * alignment_reward 
                    + progress_weight * progress_reward)

    reward_components = {
        "distance_reward": dist_reward,
        "alignment_reward": alignment_reward,
        "opening_progress_reward": progress_reward
    }
    
    return total_reward, reward_components
